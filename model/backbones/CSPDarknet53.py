import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.layers.attention_layers import SEModule, CBAM
import config.yolov4_config as cfg


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "linear": nn.Identity(),
    "mish": Mish(),
}


"""
1. 入参：filters_in（输入通道数）、filters_out（输出通道数）、kernel_size（卷积核大小）、
        stride（步长，默认为1）、norm（规范化方式，默认为"bn"）、activate（激活函数，默认为"mish"）
2. 函数功能：
   - 创建卷积层，支持规范化和激活函数
3. 返回值：x（卷积层的输出）
4. 总结：
   - Convolutional类用于创建卷积层，可以指定输入通道数、输出通道数、卷积核大小、步长、规范化方式和激活函数
   - 支持规范化方式包括"bn"（Batch Normalization），激活函数包括"leaky"、"relu"和"mish"
"""
class Convolutional(nn.Module):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size,
        stride=1,
        norm="bn",
        activate="mish",
    ):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        # 创建卷积层
        self.__conv = nn.Conv2d(
            in_channels=filters_in,
            out_channels=filters_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=not norm,
        )
        
        # 如果指定了规范化方式， 添加规范化层
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        # 如果指定了激活函数，添加激活函数层
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](
                    negative_slope=0.1, inplace=True
                )
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "mish":
                self.__activate = activate_name[activate]

    # 前向传播
    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x


"""
1. 入参：in_channels（输入通道数）、out_channels（输出通道数）、
        hidden_channels（中间隐藏层通道数，默认为None，即与输出通道数相同）、
        residual_activation（残差连接激活函数，默认为"linear"）
2. 函数功能：
   - 创建CSPBlock模块，包括两个卷积层和残差连接
   - 支持注意力机制（SEnet或CBAM）
3. 返回值：out（CSPBlock模块的输出）
4. 总结：
   - CSPBlock类用于创建CSPNet中的CSPBlock模块，支持残差连接和注意力机制
"""
class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        residual_activation="linear",
    ):
        super(CSPBlock, self).__init__()

        # 如果未指定隐藏层通道数，设为输出通道数
        if hidden_channels is None:
            hidden_channels = out_channels
        
        # 创建CSPBlock模块， 包括一个1*1卷积层和一个3*3卷积层
        self.block = nn.Sequential(
            Convolutional(in_channels, hidden_channels, 1),
            Convolutional(hidden_channels, out_channels, 3),
        )
        
        # 残差连接激活函数
        self.activation = activate_name[residual_activation]
        
        # 如果指定了注意力机制， 添加相应的注意力模块
        self.attention = cfg.ATTENTION["TYPE"]
        if self.attention == "SEnet":
            self.attention_module = SEModule(out_channels)
        elif self.attention == "CBAM":
            self.attention_module = CBAM(out_channels)
        else:
            self.attention = None

    # 前向传播
    def forward(self, x):
        residual = x
        out = self.block(x)
        
        # 如果指定了注意力机制，应用注意力模块
        if self.attention is not None:
            out = self.attention_module(out)
            
        # 残差连接
        out += residual
        return out


"""
1. 入参：in_channels（输入通道数）、out_channels（输出通道数）
2. 函数功能：
   - 创建CSPNet中的第一个阶段（CSPFirstStage），包括下采样卷积、分支卷积、块卷积和拼接卷积
   - 通过CSPBlock模块实现块卷积
3. 返回值：x（CSPFirstStage的输出）
4. 总结：
   - CSPFirstStage类用于创建CSPNet中的第一个阶段，结合了CSPBlock模块和下采样操作
"""
class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        # 下采样卷积层
        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        # 分支卷积层
        self.split_conv0 = Convolutional(out_channels, out_channels, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels, 1)

        # CSPBlock和额外卷积层组成的块
        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Convolutional(out_channels, out_channels, 1),
        )

        # 拼接卷积层
        self.concat_conv = Convolutional(out_channels * 2, out_channels, 1)

    def forward(self, x):
        # 通过下采样卷积层进行前向传播
        x = self.downsample_conv(x)
        
        # 分别通过两个分支的卷积层
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        # 通过CSPBlock和额外卷积层组成的块
        x1 = self.blocks_conv(x1)

        # 拼接两个分支的特征图
        x = torch.cat([x1, x0], dim=1)
        
        # 通过拼接卷积层进行前向传播
        x = self.concat_conv(x)

        return x


"""
1. 入参：in_channels（输入通道数）、out_channels（输出通道数）、num_blocks（块的数量）
2. 函数功能：
   - 创建CSPNet中的一个阶段（CSPStage），包括下采样卷积、分支卷积、块卷积和拼接卷积
   - 通过多个CSPBlock模块实现块卷积
3. 返回值：x（CSPStage的输出）
4. 总结：
   - CSPStage类用于创建CSPNet中的一个阶段，结合了多个CSPBlock模块和下采样操作
"""
class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        # 下采样卷积层
        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        # 分支卷积层
        self.split_conv0 = Convolutional(out_channels, out_channels // 2, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels // 2, 1)

        # 多个CSPBlock和额外卷积层组成的块
        self.blocks_conv = nn.Sequential(
            *[
                CSPBlock(out_channels // 2, out_channels // 2)
                for _ in range(num_blocks)
            ],
            Convolutional(out_channels // 2, out_channels // 2, 1)
        )

        # 拼接卷积层
        self.concat_conv = Convolutional(out_channels, out_channels, 1)

    def forward(self, x):
        # 通过下采样卷积层进行前向传播
        x = self.downsample_conv(x)

        # 分别通过两个分支的卷积层
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        # 通过多个CSPBlock和额外卷积层组成的块
        x1 = self.blocks_conv(x1)

        # 拼接两个分支的特征图
        x = torch.cat([x0, x1], dim=1)
        
        # 通过拼接卷积层进行前向传播
        x = self.concat_conv(x)

        return x


"""
1. 入参：stem_channels（干蒂通道数）、feature_channels（特征通道数列表）、num_features（输出特征的数量）、weight_path（权重文件路径）、resume（是否继续训练）
2. 函数功能：
   - 创建CSPDarknet53模型，包括干蒂卷积、多个CSP阶段（包括CSPFirstStage和多个CSPStage），最后通过这些阶段得到多个输出特征
   - 初始化权重，可选择加载预训练权重
3. 返回值：features（模型的输出特征）
4. 总结：
   - CSPDarknet53类用于构建CSPDarknet53模型，其中包含一系列CSP阶段，每个阶段包括若干CSPBlock模块
"""
class CSPDarknet53(nn.Module):
    def __init__(
        self,
        stem_channels=32,
        feature_channels=[64, 128, 256, 512, 1024],
        num_features=3,
        weight_path=None,
        resume=False,
    ):
        super(CSPDarknet53, self).__init__()

        # stem卷积
        self.stem_conv = Convolutional(3, stem_channels, 3)
        
        # 多个CSP阶段
        self.stages = nn.ModuleList(
            [
                CSPFirstStage(stem_channels, feature_channels[0]),
                CSPStage(feature_channels[0], feature_channels[1], 2),
                CSPStage(feature_channels[1], feature_channels[2], 8),
                CSPStage(feature_channels[2], feature_channels[3], 8),
                CSPStage(feature_channels[3], feature_channels[4], 4),
            ]
        )

        self.feature_channels = feature_channels
        self.num_features = num_features

        # 初始化权重
        if weight_path and not resume:
            self.load_CSPdarknet_weights(weight_path)
        else:
            self._initialize_weights()
            
    # 前向传播
    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features[-self.num_features :]

    # 初始化权重
    def _initialize_weights(self):
        # 打印初始化权重的提示信息
        print("**" * 10, "Initing CSPDarknet53 weights", "**" * 10)

        # 对模型的每个模块进行遍历
        for m in self.modules():
            # 卷积层初始化
            if isinstance(m, nn.Conv2d):
                # h*w*c
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))
                
            # Bn层初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

    # 加载CSPDarknet预训练权重
    def load_CSPdarknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, "rb") as f:
            # 从文件中读取5个3位数浮点数（np.int32）并将其忽略
            _ = np.fromfile(f, dtype=np.int32, count=5)
            # 度的去剩余的32位浮点数（np.float32）作为权重数据
            weights = np.fromfile(f, dtype=np.float32)
            
        # 初始化计数器和指针
        count = 0
        ptr = 0
        
        # 遍历模型的每个模块
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                # if count == cutoff:
                #     break
                # count += 1

                # 获取Convolutional模块中的卷积层
                conv_layer = m._Convolutional__conv
                
                
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    # 如果使用BatchNorm,加载BN层的偏置、权重、均值和方差
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias.data
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight.data
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # 如果没有BatchNorm,加载卷积层的偏置参数
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    
                # 加载卷积层的权重参数
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight.data
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


def _BuildCSPDarknet53(weight_path, resume):
    model = CSPDarknet53(weight_path=weight_path, resume=resume)

    return model, model.feature_channels[-3:]


if __name__ == "__main__":
    model = CSPDarknet53()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
