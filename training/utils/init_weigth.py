import torch.nn as nn
import torch.nn.init as init

def init_weigth(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if m.weight is not None:
            init.ones_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias)
