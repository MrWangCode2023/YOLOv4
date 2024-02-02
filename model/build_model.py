import sys

sys.path.append("..")  # 这行代码将父目录（上一级目录）添加到Python模块搜索路径中，以便在当前脚本中导入父目录中的模块或包。

import torch.nn as nn
import torch
from model.head.yolo_head import Yolo_head
from model.YOLOv4 import YOLOv4
import config.yolov4_config as cfg


"""
1. 入参：
   - `weight_path`（数据类型：str，默认值：None）：模型权重文件的路径
   - `resume`（数据类型：bool，默认值：False）：是否恢复训练的标志
   - `showatt`（数据类型：bool，默认值：False）：是否显示注意力图的标志

2. 函数功能：
   - 创建 YOLO 模型的主体部分，包括 YOLOv4 模型和 YOLO 头部（small、medium、large）
   - 根据配置信息初始化模型参数

3. 返回值：无

4. 总结：
   该类用于构建 YOLO 模型，包括 YOLOv4 模型和对应的头部（small、medium、large）。
   初始化时根据输入参数配置模型的权重文件路径、是否恢复训练以及是否显示注意力图。
   类内部包含 YOLOv4 模型和三个 YOLO 头部（small、medium、large）。
   注意：在初始化时，定义模块的顺序很重要，因为权重文件的顺序与定义的模块顺序相关。
"""

class Build_Model(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, weight_path=None, resume=False, showatt=False):
        super(Build_Model, self).__init__()
        # 是否显示注意力图
        self.__showatt = showatt
        
        # yolo模型的锚框和步长
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        
        # 类别数量
        if cfg.TRAIN["DATA_TYPE"] == "VOC":
            self.__nC = cfg.VOC_DATA["NUM"]  # {"NUM": 20，"classes": []}
        elif cfg.TRAIN["DATA_TYPE"] == "COCO":
            self.__nC = cfg.COCO_DATA["NUM"]  # {"NUM": 80，"classes": []}
        else:
            self.__nC = cfg.Customer_DATA["NUM"]
            
        # 输出通道数
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        # 创建YOLOv4模型Backbone部分
        self.__yolov4 = YOLOv4(
            weight_path=weight_path,
            out_channels=self.__out_channel,
            resume=resume,
            showatt=showatt
        )
        
        # 创建YOLO头部（small, medium, large）
        # small
        self.__head_s = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0]
        )
        # medium
        self.__head_m = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1]
        )
        # large
        self.__head_l = Yolo_head(
            nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2]
        )

    def forward(self, x):
        out = []  # 存储头部的输出
        # 获得主干部分的三个尺度的输出特征图
        [x_s, x_m, x_l], atten = self.__yolov4(x)

        # 将不同尺度的特征图通过相应的头部进行处理
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            if self.__showatt:
                return p, torch.cat(p_d, 0), atten
            return p, torch.cat(p_d, 0)


if __name__ == "__main__":
    # 导入获取模型复杂度的函数
    from utils.flops_counter import get_model_complexity_info

    # 创建YOLOv4模型实例
    net = Build_Model()
    print(net)

    # 随机生成输入图像
    in_img = torch.randn(1, 3, 416, 416)
    
    # 进行模型前向传播
    p, p_d = net(in_img)
    
    # 计算模型复杂度
    flops, params = get_model_complexity_info(
        net, (224, 224), as_strings=False, print_per_layer_stat=False
    )
    
    # 打印模型复杂度信息
    print("GFlops: %.3fG" % (flops / 1e9))
    print("Params: %.2fM" % (params / 1e6))
    
    # 打印每个尺度的输出形状
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
