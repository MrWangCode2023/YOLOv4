import numpy as np


class CosineDecayLR(object):
    """这函数实现了一个余弦衰减学习率调度器，用于在训练过程中动态地调整学习率。主要功能包括：
    1. **初始化：**
       - 接受优化器、最大步数(`T_max`)、初始学习率(`lr_init`)、最小学习率(`lr_min`)和热身步数(`warmup`)作为参数。
       - 设置调度器的初始参数。
    2. **余弦衰减：**
       - 根据余弦函数的形状，在训练过程中动态地调整学习率。
       - 衰减的步数由 `T_max` 决定，即总的训练步数。
       - 初始学习率为 `lr_init`，最小学习率为 `lr_min`。
    3. **热身阶段：**
       - 在训练开始时，通过热身步数(`warmup`)，平滑地将学习率从 0 增加到初始学习率(`lr_init`)。
       - 如果热身步数为 0，则不进行热身。
    4. **与优化器结合：**
       - 通过接受的优化器，动态地更新优化器中的学习率。
    该调度器的设计旨在在训练的不同阶段实现动态学习率调整，以提高模型的训练效果。"""
    def __init__(self, optimizer, T_max, lr_init, lr_min=0.0, warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup

    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (
                1 + np.cos(t / T_max * np.pi)
            )
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from model.YOLOv4 import YOLOv4
    import torch.optim as optim

    net = YOLOv4()
    optimizer = optim.SGD(net.parameters(), 1e-4, 0.9, weight_decay=0.0005)
    scheduler = CosineDecayLR(optimizer, 50 * 2068, 1e-4, 1e-6, 2 * 2068)

    # Plot lr schedule
    y = []
    for t in range(50):
        for i in range(2068):
            scheduler.step(2068 * t + i)
            y.append(optimizer.param_groups[0]["lr"])

    print(y)
    plt.figure()
    plt.plot(y, label="LambdaLR")
    plt.xlabel("steps")
    plt.ylabel("LR")
    plt.tight_layout()
    plt.savefig("../data/lr.png", dpi=300)
