import torch
import torch.nn as nn
import torch.nn.functional as F
import config.yolov4_config as cfg
from .backbones.CSPDarknet53 import _BuildCSPDarknet53
from .backbones.mobilenetv2 import _BuildMobilenetV2
from .backbones.mobilenetv3 import _BuildMobilenetV3
from .backbones.mobilenetv2_CoordAttention import _BuildMobileNetV2_CoordAttention
from .layers.global_context_block import ContextBlock2d

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


"""
空间金字塔池化
1. 入参：
   - `feature_channels`（数据类型：list）：包含不同特征层通道数的列表
   - `pool_sizes`（数据类型：list，默认值：[5, 9, 13]）：池化操作的不同尺寸列表
2. 函数功能：
   - 定义了一个空间金字塔池化（Spatial Pyramid Pooling）模块
   - 包括头部卷积（head_conv）和多尺度的最大池化操作
   - 通过将不同尺度的最大池化结果与输入特征拼接在一起，形成多尺度的特征表示
3. 返回值：
   - 返回多尺度的特征表示
4. 总结：
   该类定义了一个包含头部卷积和多尺度最大池化的空间金字塔池化模块。
   通过该模块，可以获得输入特征的多尺度表示，用于提取不同层次的信息。
"""
# 定义SPP类
class SpatialPyramidPooling(nn.Module):
    # 初始化方法，接受输入特征通道数和池化尺寸列表作为参数
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # 头部卷积序列，用于降低通道数
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
            Conv(feature_channels[-1] // 2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
        )

        # 多尺度最大池化操作，根据传入的池化尺寸列表构建池化层序列
        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        # 权重初始化
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim=1)

        return features

    # 初始化网络权重的私有方法
    def __initialize_weights(self):
        # 打印初始化头部卷积权重的消息
        print("**" * 10, "Initing head_conv weights", "**" * 10)

        # 遍历模型的每个模块
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用正态分布初始化权重，均值为0，标准差为0.01
                m.weight.data.normal_(0, 0.01)
                # 如果存在偏置项， 将偏置项初始化为0
                if m.bias is not None:
                    m.bias.data.zero_()
                # 打印初始化信息
                print("initing {}".format(m))
                
            # 如果是BatchNorm2d层
            elif isinstance(m, nn.BatchNorm2d):
                # 将权重初始化为1， 偏置项初始化为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # 打印初始化信息
                print("initing {}".format(m))


"""
1. 入参：
   - `in_channels`（数据类型：int）：输入通道数
   - `out_channels`（数据类型：int）：输出通道数
   - `scale`（数据类型：int，默认值：2）：上采样的尺度因子
2. 函数功能：
   - 定义了一个上采样（Upsample）模块
   - 包含一个卷积层和上采样操作
3. 返回值：
   - 无返回值，直接对输入进行上采样操作
4. 总结：
   该类定义了一个简单的上采样模块，通过卷积层和上采样操作，将输入特征图的尺寸增加。
"""
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        # 调用父类的初始化方法
        super(Upsample, self).__init__()
        
        # 定义上采样模块包含的卷积层和上采样操作
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            # 上采样层，它通过指定 scale_factor 来执行上采样。这里的 scale_factor 决定了上采样的倍数。
            nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        # 上采样操作的前向传播
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        # 调用父类的初始化方法
        super(Downsample, self).__init__()

        # 使用自定义的Conv类进行下采样操作
        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)


"""
1. 入参：
   - `feature_channels`（数据类型：list）：包含输入特征图通道数的列表
2. 函数功能：
   - 定义了PANet（Path Aggregation Network）模块
   - 该模块包含了上采样、下采样、卷积等操作，用于路径聚合
3. 返回值：
   - 返回一个包含三个特征图的列表：[downstream_feature3, upstream_feature4, upstream_feature5]
4. 总结：
   - PANet模块通过特征图的上下游路径聚合，生成三个特征图，用于后续的目标检测任务。
"""
class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        # 特征转换模块，用于调整输入特征的通道数
        self.feature_transform3 = Conv(
            feature_channels[0], feature_channels[0] // 2, 1
        )
        self.feature_transform4 = Conv(
            feature_channels[1], feature_channels[1] // 2, 1
        )

        # 上采样和下采样模块
        self.resample5_4 = Upsample(
            feature_channels[2] // 2, feature_channels[1] // 2
        )
        self.resample4_3 = Upsample(
            feature_channels[1] // 2, feature_channels[0] // 2
        )
        self.resample3_4 = Downsample(
            feature_channels[0] // 2, feature_channels[1] // 2
        )
        self.resample4_5 = Downsample(
            feature_channels[1] // 2, feature_channels[2] // 2
        )
        
        # 下游卷积模块
        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
        )
        
        # 上游卷积模块
        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        
        # 初始化权重
        self.__initialize_weights()

    def forward(self, features):
        features = [
            self.feature_transform3(features[0]),
            self.feature_transform4(features[1]),
            features[2],
        ]

        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(
            torch.cat(
                [features[1], self.resample5_4(downstream_feature5)], dim=1
            )
        )
        downstream_feature3 = self.downstream_conv3(
            torch.cat(
                [features[0], self.resample4_3(downstream_feature4)], dim=1
            )
        )

        upstream_feature4 = self.upstream_conv4(
            torch.cat(
                [self.resample3_4(downstream_feature3), downstream_feature4],
                dim=1,
            )
        )
        upstream_feature5 = self.upstream_conv5(
            torch.cat(
                [self.resample4_5(upstream_feature4), downstream_feature5],
                dim=1,
            )
        )

        return [downstream_feature3, upstream_feature4, upstream_feature5]

    """
    1. 入参：无
    2. 函数功能：
       - 初始化PANet模块的权重
       - 打印初始化信息
    3. 返回值：无
    4. 总结：
       - 通过遍历模块的方式，对卷积层（`nn.Conv2d`）和批归一化层（`nn.BatchNorm2d`）的权重进行初始化
       - 对卷积层，使用正态分布初始化权重，偏置项初始化为零
       - 对批归一化层，权重初始化为1，偏置项初始化为零
       - 在初始化过程中打印相关信息
    """
    def __initialize_weights(self):
        # 打印初始化权重的提示信息
        print("**" * 10, "Initing PANet weights", "**" * 10)

        # 对模块进行遍历，进行初始化
        for m in self.modules():
            # 卷积层初始化，使用正态分布初始化权重，偏执初始化为零
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # 答应初始化的信息
                print("initing {}".format(m))
                
            # 批归一化层初始化，将权重初始化为1， 偏执初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # 打印初始化信息
                print("initing {}".format(m))

"""
1. 入参：feature_channels（特征通道数列表）、target_channels（目标通道数）
2. 函数功能：
   - 创建PredictNet模块
   - 包含多个卷积层和1x1卷积层，用于产生目标预测
   - 初始化权重并打印初始化信息
3. 返回值：无
4. 总结：
   - PredictNet模块包含多个卷积层，每个卷积层包含一个3x3卷积和一个1x1卷积，用于生成目标预测
   - 初始化过程中采用正态分布初始化卷积层的权重，偏置项初始化为零
   - 批归一化层的权重初始化为1，偏置项初始化为零
   - 在初始化过程中打印相关信息
"""
class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):
        super(PredictNet, self).__init__()

        # 定义预测网络的卷积层
        self.predict_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(feature_channels[i] // 2, feature_channels[i], 3),
                    nn.Conv2d(feature_channels[i], target_channels, 1),
                )
                for i in range(len(feature_channels))
            ]
        )
        # 初始化权重
        self.__initialize_weights()

    def forward(self, features):
        # 对每个特征层进行预测
        predicts = [
            predict_conv(feature)
            for predict_conv, feature in zip(self.predict_conv, features)
        ]

        return predicts

    def __initialize_weights(self):
        # 打印初始化权重的提示信息
        print("**" * 10, "Initing PredictNet weights", "**" * 10)

        # 遍历网络模块进行初始化
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                # 权重正态分布初始化
                m.weight.data.normal_(0, 0.01)
                # 偏置初始化为0
                if m.bias is not None:
                    m.bias.data.zero_()
                # 打印初始化信息
                print("initing {}".format(m))
                
            # 批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                # 权重初始化为1
                m.weight.data.fill_(1)
                # 偏执初始化为0
                m.bias.data.zero_()
                # 打印初始化信息
                print("initing {}".format(m))


"""
1. 入参：weight_path（权重路径，默认为None）、out_channels（输出通道数，默认为255）、
        resume（是否恢复训练，默认为False）、showatt（是否显示注意力图，默认为False）、
        feature_channels（特征通道数，默认为0）
2. 函数功能：
   - 创建YOLOv4模型，包含不同类型的backbone（CSPDarknet53、MobilenetV2、CoordAttention、MobilenetV3）
   - 包含注意力模块、Spatial Pyramid Pooling、Path Aggregation Net和目标预测模块
3. 返回值：predicts（目标预测结果）、atten（注意力图，如果showatt为True）
4. 总结：
   - YOLOv4模型包含不同类型的backbone，注意力模块、Spatial Pyramid Pooling、Path Aggregation Net和目标预测模块
   - 如果showatt为True，返回注意力图atten
   - 前向传播中依次经过backbone、注意力模块（如果showatt为True）、Spatial Pyramid Pooling、Path Aggregation Net和目标预测模块
"""
class YOLOv4(nn.Module):
    def __init__(self, weight_path=None, out_channels=255, resume=False, showatt=False, feature_channels=0):
        super(YOLOv4, self).__init__()
        self.showatt = showatt
        
        # 根据配置选择不同的模型类型，并获取backbone和feature_channels
        if cfg.MODEL_TYPE["TYPE"] == "YOLOv4":
            # CSPDarknet53 backbone
            self.backbone, feature_channels = _BuildCSPDarknet53(
                weight_path=weight_path, resume=resume
            )
        elif cfg.MODEL_TYPE["TYPE"] == "Mobilenet-YOLOv4":
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobilenetV2(
                weight_path=weight_path, resume=resume
            )
        elif cfg.MODEL_TYPE["TYPE"] == "CoordAttention-YOLOv4":
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobileNetV2_CoordAttention(
                weight_path=weight_path, resume=resume
            )
        elif cfg.MODEL_TYPE["TYPE"] == "Mobilenetv3-YOLOv4":
            # MobilenetV3 backbone
            self.backbone, feature_channels = _BuildMobilenetV3(
                weight_path=weight_path, resume=resume
            )
        else:
            assert print("model type must be YOLOv4 or Mobilenet-YOLOv4")

        # 如果需要显示注意力图，则添加注意力模块
        if self.showatt:
            self.attention = ContextBlock2d(feature_channels[-1], feature_channels[-1])
            
        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(feature_channels)

        # Path Aggregation Net（PANet实例）
        self.panet = PANet(feature_channels)

        # predict：预测网络
        self.predict_net = PredictNet(feature_channels, out_channels)

    def forward(self, x):
        atten = None
        features = self.backbone(x)  # backbone部分的输出,P3， P4， P5
        
        # 如果需要显示注意力图，则在最后一个特征图上应用注意力模块
        if self.showatt:
            features[-1], atten = self.attention(features[-1])
            
        # 进行Spatial Pyramid Pooling
        features[-1] = self.spp(features[-1])  # 最后一个特征层进行空间金字塔池化
        # 进行PANet
        features = self.panet(features)
        # 进行预测
        predicts = self.predict_net(features)
        return predicts, atten


"""
1. 入参：无

2. 函数功能：
   - 在GPU或CPU上创建YOLOv4模型，将其移动到设备上
   - 生成随机输入张量，通过模型进行前向传播，打印每个输出的形状
   - 循环进行前向传播，持续打印输出的形状

3. 输出：
   - predicts[0]：downstream_feature3 的形状
   - predicts[1]：upstream_feature4 的形状
   - predicts[2]：upstream_feature5 的形状
"""
if __name__ == "__main__":
    # 检查是否支持CUDA,返回布尔值
    cuda = torch.cuda.is_available()
    # 选择设备，如果支持CUDA， 则使用第一个GPU,否则使用CPU
    device = torch.device("cuda:{}".format(0) if cuda else "cpu")
    
    # 创建YOLOv4模型并将其移动到设备上
    model = YOLOv4().to(device)
    
    # 生成随机输入张量
    x = torch.randn(1, 3, 160, 160).to(device)
    
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    
    # 前向传播，打印每个输出的形状
    while 1:
        predicts = model(x)
        print(predicts[0].shape)
        print(predicts[1].shape)
        print(predicts[2].shape)
