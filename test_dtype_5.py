import torch
import torch.nn as nn

m = nn.Conv2d(3, 32, 4, stride=2).cuda().bfloat16()
x = torch.randn(2, 3, 16, 16, dtype=torch.float32).cuda()
m(x)
