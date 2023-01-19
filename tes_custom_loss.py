
import torch

torch.manual_seed(0)


c = torch.rand(32, 3, 20)
d = torch.rand(32, 3, 20)
# r = torch.cat((c, d), dim=1)
print(c)

lis = [torch.roll(d, i, dims=2) for i in range(0, d.shape[2])]

#x = torch.tensor(lis)
x = torch.stack(lis, dim=0)
# print(x.shape)
# print(x[0])
# print(x[1])
k = (x - c).pow(2).sum(dim=(1, 2, 3))
i = torch.argmin(k)
print(i)
answer = x[i]
print(answer.shape)
