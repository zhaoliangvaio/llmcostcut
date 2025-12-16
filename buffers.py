import torch, time, random
from collections import deque
class RingBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.size = 0
        # list of dicts
        self.data = [
            {
                "text": None,
                "encoding": None,
                "label": None,
                "student_pred": None,
                "confidence": None,
                "timestamp": None,
            }
            for _ in range(capacity)
        ]
    def add(self, **kwargs):
        i = self.ptr

        for k, v in kwargs.items():
            if k == "text":
                # text must stay Python str
                self.data[i][k] = v
            else:
                # convert everything else to tensor
                self.data[i][k] = torch.as_tensor(v)

        self.data[i]["timestamp"] = torch.tensor(time.time(), dtype=torch.float32)

        # update pointer + size
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True
        self.size = min(self.size + 1, self.capacity)


    # def sample(self, batch=64):
    #     maxidx = self.capacity if self.full else self.ptr
    #     idxs = np.random.randint(0, maxidx, batch)
    
    #     # return a list of dicts
    #     return [self.data[i] for i in idxs]

    def sample(self, batch=64, num_labels=4):
        """Balanced sampling across classes."""

        if self.size == 0:
            return []

        # Identify valid indices
        maxidx = self.size if not self.full else self.capacity

        # Group indices by class
        class_to_indices = {c: [] for c in range(num_labels)}
        for i in range(maxidx):
            lbl = self.data[i]["label"]
            if lbl is not None:
                lbl = int(lbl.item()) if torch.is_tensor(lbl) else int(lbl)
                if lbl in class_to_indices:
                    class_to_indices[lbl].append(i)

        # Compute per-class sample quota
        per_class = batch // num_labels

        sampled_indices = []

        for c in range(num_labels):
            idx_list = class_to_indices[c]
            if len(idx_list) == 0:
                # class missing → sample from whole pool
                fallback = random.choices(range(maxidx), k=per_class)
                sampled_indices.extend(fallback)
            else:
                # oversample with replacement
                sampled = random.choices(idx_list, k=per_class)
                sampled_indices.extend(sampled)

        # If rounding left some space, fill randomly
        while len(sampled_indices) < batch:
            sampled_indices.append(random.randrange(maxidx))

        # Return list of dicts
        return [self.data[i] for i in sampled_indices]




class ReplayBufferManager:
    def __init__(self, per_buffer_capacity=20000):
        self.buffers = {}
        self.capacity = per_buffer_capacity

    def get_buffer(self, workflow_id):
        if workflow_id not in self.buffers:
            self.buffers[workflow_id] = RingBuffer(self.capacity)
        return self.buffers[workflow_id]

    def add_sample(self, workflow_id, text, encoding, label, student_pred=None, confidence=None):
        buf = self.get_buffer(workflow_id)
        buf.add(
            text=text,
            encoding=encoding,
            label=label,
            student_pred=student_pred,
            confidence=confidence
        )

    def sample_for_training(self, workflow_id, batch_size=64):
        print("batch_size_old:",batch_size)
        if(batch_size>self.buffers[workflow_id].size):
            batch_size = self.buffers[workflow_id].size
        print("batch_size_new:",batch_size)
        return self.buffers[workflow_id].sample(batch_size)