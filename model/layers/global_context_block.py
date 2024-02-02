import torch
from torch import nn
from mmcv.cnn import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        # 如果是Sequential模块， 将最后一层的权重初始化为0
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        # 否则， 将当前层的权重初始化为0
        constant_init(m, val=0)
        m.inited = True

"""
这是 `ContextBlock2d` 类的完整代码，下面是对该类的功能的简要总结：
### 输入(Input):
- **形状(Shape):** `(batch, channels, height, width)`
- **类型(Type):** Torch Tensor
- **实际意义(Meaning):** 输入特征图，其中：
  - `batch` 表示批处理的图像数量。
  - `channels` 表示图像的通道数。
  - `height` 表示图像的高度。
  - `width` 表示图像的宽度。
### 初始化(Initialization):
- **形状(Shape):** 无
- **类型(Type):** 无
- **实际意义(Meaning):** 初始化卷积核权重和通道注意力模块的权重。
### 通道注意力计算(Spatial Channel Attention Computation):
- **形状(Shape):** `(batch, 1, height, width)`
- **类型(Type):** Torch Tensor
- **实际意义(Meaning):** 计算每个通道的注意力权重，通过卷积和 Softmax 运算得到。
### 前向传播(Forward Pass):
- **形状(Shape):** `(batch, channels, height, width)`
- **类型(Type):** Torch Tensor
- **实际意义(Meaning):** 通过通道注意力计算得到的权重，对输入特征进行加权，然后经过一系列卷积、归一化和激活函数操作。
### 输出(Output):
- **形状(Shape):** `(batch, channels, height, width)`
- **类型(Type):** Torch Tensor
- **实际意义(Meaning):** 输出特征图，表示经过通道注意力和一系列操作后的特征。
- **形状(Shape):** `(batch, 1, height, height)`
- **类型(Type):** Torch Tensor
- **实际意义(Meaning):** 注意力矩阵，表示通道间的注意力关系。
这个模块的主要目的是通过通道注意力机制增强输入特征图的表达能力。如果您有其他问题或需要更多的解释，请随时告诉我。
"""
# 定义一个上下文块的类
class ContextBlock2d(nn.Module):
    # 初始化方法，接受输入通道数和输出通道数作为参数
    def __init__(self, inplanes, planes):
        super(ContextBlock2d, self).__init__()
        # 设置输入通道数和输出通道数
        self.inplanes = inplanes
        self.planes = planes
        # 1*1卷积用于生成注意力权重
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        # 对生成的权重进行归一化
        self.softmax = nn.Softmax(dim=2)
        # 通道注意力调整的卷积序列
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
        )
        # 初始化模块的参数
        self.reset_parameters()

    # 初始化权重的方法
    def reset_parameters(self):
        # 使用kaiming初始化1*1卷积层的权重
        kaiming_init(self.conv_mask, mode="fan_in")
        self.conv_mask.inited = True
        # 使用last_zeros_init初始化通道注意力调整的卷积序列的权重
        last_zero_init(self.channel_add_conv)

    # 空间池化方法，用于生成通道注意力的权重和注意力矩阵
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # 将输入特征图展平成二维张量[N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # 在通道维度上添加一个维度[N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # 用1*1卷积生成注意力权重[N, 1, H, W]
        context_mask = self.conv_mask(x)
        # 将生成的权重展平成二维张量[N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # 权重进行softmask归一化，得到两个注意力权重矩阵[N, 1, H * W]
        context_mask = self.softmax(context_mask)
        beta1 = context_mask
        beta2 = torch.transpose(beta1, 1, 2)
        atten = torch.matmul(beta2, beta1)

        # 添加维度[N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # 使用注意力权重矩阵对输入进行加权，得到context[N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # 将context展平成[N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context, atten

    # 前向传播方法
    def forward(self, x):
        # 获取空间池化生成的通道注意力和注意力矩阵[N, C, 1, 1]
        context, atten = self.spatial_pool(x)
        # 将通道注意力调整的卷积序列应用到输入特征上[N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        # 将最后的特征与原始特征相加，得到最终输出
        out = x + channel_add_term

        return out, atten
