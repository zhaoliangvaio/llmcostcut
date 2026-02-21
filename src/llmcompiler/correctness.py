"""
LLMCompiler: Adaptive LLM-to-Small-Model Distillation Framework

Copyright (c) 2024–2025
Liang Zhao and Ziyang Yu
Emory University

This file is part of the LLMCompiler framework.
Released under the Apache 2.0 License (see LICENSE).

If you use this code in academic work, please cite:
<Paper citation to appear>

Contact:
Liang Zhao (liang.zhao@emory.edu)
"""
import torch, math
import threading
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

_CUDA_INIT_LOCK = threading.Lock()
_CUDA_READY = False
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
        self._ensure_cuda_ready_once()
        # Serialize predictor state/model access across threads.
        self._state_lock = threading.Lock()

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
        self.enc_mean = torch.zeros(encoding_dim, device=self.device)
        self.enc_cov  = torch.eye(encoding_dim, device=self.device)
        self._eye = torch.eye(encoding_dim, device=self.device)
        self.n_total  = 0
        self._last_p_correct = 0.5

    def _ensure_cuda_ready_once(self):
        global _CUDA_READY
        if self.device.type != "cuda":
            return
        with _CUDA_INIT_LOCK:
            if _CUDA_READY:
                return
            # Pre-initialize CUDA context once in a thread-safe way.
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize(self.device)
            _CUDA_READY = True

    def _knn_dist(self, encoding, memory, k=5):
        if len(memory) == 0:
            return torch.tensor(0.0, device=self.device)
        memory_tensor = torch.stack(list(memory)).to(encoding.device)
        dists = torch.norm(encoding.unsqueeze(0) - memory_tensor, dim=1)
        k = min(k, len(dists))
        return dists.topk(k, largest=False).values.mean()
    
    def _mahalanobis(self, encoding):
        # Keep this path numerically/thread stable under heavy concurrency.
        # With current defaults enc_cov is identity (not updated online), so
        # Mahalanobis distance reduces to Euclidean norm around enc_mean.
        diff = encoding - self.enc_mean
        return torch.norm(diff, p=2)

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
        ], dtype=torch.float32, device=self.device)

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
        with self._state_lock:
            correct = int(teacher_label == student_label)
            feats = self._build_features(encoding, student_logits, input_len)

            self.buffer_feats.append(feats.detach().cpu())
            self.buffer_labels.append(correct)

            self._update_prototypes(encoding.to(self.device), correct)
            if correct:
                self.recent_correct.append(encoding.detach().cpu())
            else:
                self.recent_wrong.append(encoding.detach().cpu())


    def train_step(self, batch=64, steps=1):
        if not self._state_lock.acquire(blocking=False):
            return
        try:
            if len(self.buffer_feats) < batch:
                return

            for _ in range(steps):
                idxs = torch.randint(0, len(self.buffer_feats), (batch,), device="cpu").tolist()
                feats = torch.stack([self.buffer_feats[i] for i in idxs]).to(self.device)
                labels = torch.tensor([self.buffer_labels[i] for i in idxs],
                                      dtype=torch.float32, device=self.device)

                preds = self.model(feats)
                loss = F.binary_cross_entropy(preds, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        finally:
            self._state_lock.release()

    @torch.no_grad()
    def predict_p_correct(self, encoding, student_logits, input_len):
        if not self._state_lock.acquire(blocking=False):
            probs = F.softmax(student_logits.to(self.device), dim=-1)
            return float(probs.max().item())
        try:
            self.model.eval()
            feats = self._build_features(encoding, student_logits, input_len).to(self.device)
            p = float(self.model(feats.unsqueeze(0))[0])
            self._last_p_correct = p
            return p
        finally:
            self._state_lock.release()