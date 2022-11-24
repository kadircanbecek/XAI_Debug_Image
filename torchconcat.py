import torch
from torch import nn
from torch.nn import MSELoss


class M1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(10, 5)

    def forward(self, inp):
        return self.l(inp)


class M2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(5, 1)

    def forward(self, inp):
        return self.l(inp)
loss_fn = MSELoss()

m1 = M1()
m2 = M2()
i = torch.rand(1, 10)
o1 = m1(i)
o2 = m2(o1)
loss=loss_fn(o2,torch.tensor(0,dtype=torch.float))
loss.backward()
for name, param in m1.named_parameters():
    print(name, param.grad)

for name, param in m2.named_parameters():
    print(name, param.grad)
pass
