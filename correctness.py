import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
class CorrectnessMLP(nn.Module):
    def __init__(self, in_dim=7, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

class OnlineCorrectnessPredictor:
    """
    This module:
    1) Builds feature vector from (encoding, logits, length, prototypes)
    2) Maintains a replay buffer of (features, correct_label)
    3) Supports online incremental training
    4) Produces correctness probability 0–1
    """

    def __init__(self, encoding_dim, num_classes, hidden=32, device="cpu"):
        self.device = torch.device(device)

        self.encoding_dim = encoding_dim
        self.num_classes = num_classes

        # small: 10 features
        self.in_dim = 10

        self.model = CorrectnessMLP(self.in_dim, hidden).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # replay memory
        self.buffer_feats = deque(maxlen=5000)
        self.buffer_labels = deque(maxlen=5000)

        # prototypes
        self.n_correct = 0
        self.n_wrong = 0
        self.proto_correct = torch.zeros(encoding_dim).to(self.device)
        self.proto_wrong = torch.zeros(encoding_dim).to(self.device)
        self.recent_correct = deque(maxlen=2000)
        self.recent_wrong   = deque(maxlen=2000)

        # for Mahalanobis
        self.enc_mean = torch.zeros(encoding_dim)
        self.enc_cov  = torch.eye(encoding_dim)
        self.n_total  = 0
    def _knn_dist(self, encoding, memory, k=5):
        if len(memory) == 0:
            return torch.tensor(0.0, device=self.device)
        dists = torch.stack([torch.norm(encoding - m) for m in memory])
        k = min(k, len(dists))
        return dists.topk(k, largest=False).values.mean()
    def _mahalanobis(self, encoding):
        diff = encoding - self.enc_mean.to(self.device)
        inv_cov = torch.inverse(self.enc_cov.to(self.device) + 1e-6 * torch.eye(self.enc_cov.size(0)))
        return torch.sqrt(diff @ inv_cov @ diff)

    # ---------------- Feature Engineering ----------------
    def _build_features(self, encoding, student_logits, input_len):
        """
        Returns torch.Tensor [7]
        """
        encoding = encoding.to(self.device)
        logits = student_logits.to(self.device)
        probs = F.softmax(logits, dim=-1)

        p_max = probs.max()
        top2 = probs.topk(2).values
        margin = top2[0] - top2[1]
        entropy = -(probs * (probs + 1e-12).log()).sum()

        enc_norm = encoding.norm()
        len_norm = math.log(1 + input_len)

        # prototype distances
        if self.n_correct > 0:
            dist_correct = (encoding - self.proto_correct).norm()
        else:
            dist_correct = torch.tensor(0.0, device=self.device)

        if self.n_wrong > 0:
            dist_wrong = (encoding - self.proto_wrong).norm()
        else:
            dist_wrong = torch.tensor(0.0, device=self.device)
        knn_correct = self._knn_dist(encoding, self.recent_correct)
        knn_wrong   = self._knn_dist(encoding, self.recent_wrong)
        mahal       = self._mahalanobis(encoding)

        feat = torch.tensor([
            p_max, margin, entropy, enc_norm.item(), len_norm,
            dist_correct.item(), dist_wrong.item(),
            knn_correct.item(), knn_wrong.item(), mahal.item()
        ])

        # feat = torch.tensor([
        #     p_max.item(),
        #     margin.item(),
        #     entropy.item(),
        #     enc_norm.item(),
        #     len_norm,
        #     dist_correct.item(),
        #     dist_wrong.item()
        # ], device=self.device)

        return feat

    # ---------------- Prototype Updating ----------------
    def _update_prototypes(self, enc, correct):
        if correct:
            self.n_correct += 1
            eta = 1.0 / self.n_correct
            self.proto_correct = (1 - eta) * self.proto_correct + eta * enc
        else:
            self.n_wrong += 1
            eta = 1.0 / self.n_wrong
            self.proto_wrong = (1 - eta) * self.proto_wrong + eta * enc

    # ---------------- Public API ----------------
    def add_training_example(self, encoding, student_logits, input_len, teacher_label, student_label):
        """
        Only called when you QUERY the teacher.
        """
        correct = int(teacher_label == student_label)
        feats = self._build_features(encoding, student_logits, input_len)

        self.buffer_feats.append(feats)
        self.buffer_labels.append(correct)

        self._update_prototypes(encoding, correct)
        if correct:
            self.recent_correct.append(encoding.detach().cpu())
        else:
            self.recent_wrong.append(encoding.detach().cpu())


    def train_step(self, batch=64, steps=1):
        if len(self.buffer_feats) < batch:
            return

        for _ in range(steps):
            idxs = torch.randint(0, len(self.buffer_feats), (batch,))
            feats = torch.stack([self.buffer_feats[i] for i in idxs]).to(self.device)
            labels = torch.tensor([self.buffer_labels[i] for i in idxs],
                                  dtype=torch.float32, device=self.device)

            preds = self.model(feats)
            loss = F.binary_cross_entropy(preds, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    @torch.no_grad()
    def predict_p_correct(self, encoding, student_logits, input_len):
        self.model.eval()
        feats = self._build_features(encoding, student_logits, input_len)
        return float(self.model(feats.unsqueeze(0))[0])