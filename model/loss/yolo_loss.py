import sys

sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import config.yolov4_config as cfg


# 焦点损失函数类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        # 初始化方法，定义Focal Loss的超参数和使用的二分类交叉熵损失
        super(FocalLoss, self).__init__()
        # 设置焦点损失的调制因子
        self.__gamma = gamma
        # 创建二分类交叉熵损失对象，用于计算损失
        self.__alpha = alpha
        # 创建二分类交叉熵损失对象，用于计算损失
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    # 前向传播方法，计算Focal Loss
    def forward(self, input, target):
        # 计算二分类交叉熵损失
        loss = self.__loss(input=input, target=target)
        # 对损失进行修正，乘以权重因子
        loss *= self.__alpha * torch.pow(
            torch.abs(target - torch.sigmoid(input)), self.__gamma
        )  # torch.pow（x, y）用于进行逐元素指数运算:x的y次方

        return loss

#
class YoloV4Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        # 调用父类（nn.Module）的构造函数
        super(YoloV4Loss, self).__init__()
        
        # 初始化用于损失计算的锚点、步长和IOU阈值的私有属性
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides

    def forward(
        self,
        p,
        p_d,
        label_sbbox,
        label_mbbox,
        label_lbbox,
        sbboxes,
        mbboxes,
        lbboxes,
    ):
        """
        :param p: 三个检测层的预测偏移值
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
                    
        :param p_d: 解码后的预测值，值的大小适用于图像大小
                    例如  p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
                    
        :param label_sbbox:小型检测层的标签，值的大小适用于原始图像大小，
                            形状为[bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        
        :param label_mbbox: 中型检测层的标签，值的大小适用于原始图像大小。
                            形状为[bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        
        :param label_lbbox: 大型检测层的标签，值的大小适用于原始图像大小。
                            形状为[bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        
        :param sbboxes: 小型检测层的边界框，值的大小适用于原始图像大小。
                        形状为[bs, 150, x+y+w+h]
                        
        :param mbboxes: 中型检测层的边界框，值的大小适用于原始图像大小。
                        形状为[bs, 150, x+y+w+h]
        
        :param lbboxes: 大型检测层的边界框，值的大小适用于原始图像大小。
                        形状为[bs, 150, x+y+w+h]
        """
        
        # 提取每个检测层的步长
        strides = self.__strides

        # 计算小型检测层的损失
        (
            loss_s,
            loss_s_ciou,
            loss_s_conf,
            loss_s_cls,
        ) = self.__cal_loss_per_layer(
            p[0], p_d[0], label_sbbox, sbboxes, strides[0]
        )
        
        # 计算中型检测层的损失
        (
            loss_m,
            loss_m_ciou,
            loss_m_conf,
            loss_m_cls,
        ) = self.__cal_loss_per_layer(
            p[1], p_d[1], label_mbbox, mbboxes, strides[1]
        )
        
        # 计算大型检测层的损失
        (
            loss_l,
            loss_l_ciou,
            loss_l_conf,
            loss_l_cls,
        ) = self.__cal_loss_per_layer(
            p[2], p_d[2], label_lbbox, lbboxes, strides[2]
        )

        # 汇总所有检测层损失
        loss = loss_l + loss_m + loss_s
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_ciou, loss_conf, loss_cls

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
         (1) 目标框回归损失。
            使用GIOU损失，详见 https://arxiv.org/abs/1902.09630。
            注意: 损失因子为2-w*h/(img_size**2)，用于影响不同尺度上损失值的平衡。
             balance of the loss value at different scales.
             
        (2)置信度损失。
        包括前景和背景的置信度损失。
        注意: 当由特征点预测的框与所有GT的最大IoU小于阈值时，计算背景损失。

        (3)类别损失。
        类别损失采用二元交叉熵（BCE），对每个类别使用二进制值。
        :param stride: 特征图相对于原始图像的尺度比例
        :return: 该检测层所有批次的平均损失(loss_giou, loss_conf, loss_cls)
        """
        
        # 二元交叉熵损失函数
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        
        # Focal Loss损失函数
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        # 获取批次大小和特征图网格大小
        batch_size, grid = p.shape[:2]
        
        # 计算特征图相对于原始图像的尺度
        img_size = stride * grid

        # 提取预测值中的置信度和类别信息
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        # 提取解码后的预测值中的坐标信息
        p_d_xywh = p_d[..., :4]

        # 提取标签中的坐标、目标掩码、类别信息以及混合信息
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # 计算CIou损失
        ciou = tools.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # 缩放的bbox权重用于平衡小物体和大物体对损失的影像
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[
            ..., 3:4
        ] / (img_size ** 2)
        
        # 计算CIou损失
        loss_ciou = label_obj_mask * bbox_loss_scale * (1.0 - ciou) * label_mix

        # 计算预测框与所有GT框的IoU
        iou = tools.CIOU_xywh_torch(
            p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )
        
        # 获取IoU的最大值
        iou_max = iou.max(-1, keepdim=True)[0]
        
        # 计算背景目标掩码
        label_noobj_mask = (1.0 - label_obj_mask) * (
            iou_max < self.__iou_threshold_loss
        ).float()

        # 计算置信度损失
        loss_conf = (
            label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask)
            + label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)
        ) * label_mix

        # 计算类别损失
        loss_cls = (
            label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix
        )

        loss_ciou = (torch.sum(loss_ciou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_ciou + loss_conf + loss_cls

        return loss, loss_ciou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.build_model import Yolov4

    # 创建YOLOv4模型
    net = Yolov4()

    # torch.rand(3, 3, 416, 416)生成随机输入数据
    # 获取YOLOv4的输出
    p, p_d = net(torch.rand(3, 3, 416, 416))
    
    # 生成随机标签和边界框
    label_sbbox = torch.rand(3, 52, 52, 3, 26)
    label_mbbox = torch.rand(3, 26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3, 26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    # 计算YOLOv4损失
    loss, loss_xywh, loss_conf, loss_cls = YoloV4Loss(
        cfg.MODEL["ANCHORS"], cfg.MODEL["STRIDES"]
    )(p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    
    # 打印总损失
    print(loss)
