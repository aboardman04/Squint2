import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 4, stride=2).cuda()

    def forward(self, x):
        return self.conv(x)

m = Model()
x = torch.randn(2, 3, 16, 16).cuda()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = m(x)
print("SUCCESS!")
