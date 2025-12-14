import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedulr(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
