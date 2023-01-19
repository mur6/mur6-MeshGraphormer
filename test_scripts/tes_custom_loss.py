
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
import matplotlib.pyplot as plt

def main():
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


def scheduler():
    model = torch.nn.Linear(1, 1) ## 適当なモデル
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    num_epochs = 500

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=40,
        cycle_limit=11,
        cycle_decay=0.8,
        lr_min=0.0001,
        warmup_t=20,
        warmup_lr_init=5e-5,
        warmup_prefix=True)

    lrs = []
    for i in range(num_epochs):
        lrs.append(scheduler.get_epoch_values(i))
    plt.plot(lrs)
    plt.show()

if __name__ == "__main__":
    scheduler()
