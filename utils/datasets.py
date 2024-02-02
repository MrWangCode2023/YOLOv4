"""
1. 总结：
   该代码定义了一个自定义数据集类（Build_Dataset），用于目标检测模型的训练。同时，展示了数据增强的过程和标签的创建。

2. 输入：
   - `anno_file_type`：数据集的类型，可以是"train"或者"test"。
   - `img_size`：图片的尺寸，默认为416。

3. 功能：
   - 数据集初始化：
     - 通过指定数据集类型加载相应的注释文件。
     - 定义类别、类别数、类别到ID的映射关系。
   - 数据集获取：
     - 根据给定索引，解析注释文件，获取原始图像和边界框。
     - 随机选择另一个图像，进行Mixup数据增强。
     - 对增强后的图像和边界框进行标签创建。
   - 数据集长度获取。

4. 输出：
   返回训练时需要的图像和标签，包括小、中、大尺度的标签，以及对应的边界框。返回的图像和标签可以直接用于目标检测模型的训练。
"""




# coding=utf-8
import os
import sys

sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov4_config as cfg
import cv2
import numpy as np
import random

# from . import data_augment as dataAug
# from . import tools

import utils.data_augment as dataAug
import utils.tools as tools



"""
class Build_Dataset(Dataset):
1. **输入**:
   - 形状：原始图像数据的形状为 (H, W, C)，其中 H 为高度，W 为宽度，C 为通道数。
   - 类型：原始边界框注释数据的类型为列表，每个元素包含图像路径、边界框坐标和类别。
2. **算法流程**:
   - **图像数据处理**:
     1. 读取原始图像，确保图像存在。
     2. 将边界框坐标信息转换为 NumPy 数组。
     3. 应用一系列数据增强操作：
        - 随机水平翻转。
        - 随机裁剪。
        - 随机仿射变换。
        - 调整大小至指定的图像尺寸。
   - **Mixup 操作**:
     1. 随机选择另一个样本进行 Mixup 操作。
     2. 对选定样本进行图像通道的转置操作。
   - **标签生成**:
     1. 根据边界框信息，计算与模型锚点的匹配关系。
     2. 生成训练标签数组，包括目标存在的概率、边界框坐标、类别信息等。
     3. 处理同一目标可能匹配到多个锚点或多个目标可能匹配到同一锚点的情况。
   - **返回处理后的数据**:
     1. 将处理后的图像和标签数据转换为 PyTorch 的 Tensor 格式。
3. **输出**:
   - 形状：处理后的图像数据形状为 (C, H, W)，其中 C 为通道数，H 为图像高度，W 为图像宽度。
   - 类型：处理后的标签数据类型为 PyTorch 的 Tensor，包括图像标签和边界框坐标。
通过这一流程，数据增强算法能够提高模型对多样性数据的学习能力，增强模型的泛化性能。
"""
class Build_Dataset(Dataset):
    def __init__(self, anno_file_type, img_size=416):
        # 初始化数据集对象
        self.img_size = img_size  # 图像尺寸用于多尺度训练
        if cfg.TRAIN["DATA_TYPE"] == "VOC":
            self.classes = cfg.VOC_DATA["CLASSES"]  # 获取VOC数据集类别名称
        elif cfg.TRAIN["DATA_TYPE"] == "COCO":
            self.classes = cfg.COCO_DATA["CLASSES"]  # 获取COCO数据集类别名称
        else:
            self.classes = cfg.Customer_DATA["CLASSES"]  # 获取自定义数据集名称
        self.num_classes = len(self.classes)  # 获取数据集类别数量
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))  # {class: class_num; ...}
        # 加载注释信息
        self.__annotations = self.__load_annotations(anno_file_type)  # 加载指定类型的注释数据

    def __len__(self):
        # 返回注释列表长度
        return len(self.__annotations)

    """
    1. 总结：
       - 该函数用于获取数据集中指定索引位置的图像和标签数据，包括图像的 Mixup 处理和生成相应的标签。
       - 通过调用私有方法 `__parse_annotation` 解析原始图像和边界框注释，同时进行图像通道的转置操作。
       - 随机选择另一个索引，获取对应图像并进行 Mixup 操作，生成混合后的图像和边界框。
       - 调用私有方法 `__creat_label` 生成训练标签和边界框坐标。
       - 将处理后的图像和标签数据转换为 PyTorch 的 Tensor 格式并返回。
    2. 输入：
       - item: 数据集中指定索引位置的样本序号。
    3. 功能：
       - 获取原始图像和边界框注释，进行 Mixup 操作，生成训练所需的图像和边界框。
       - 调用 `__creat_label` 方法生成对应的训练标签和边界框坐标。
    
    4. 输出：
       - img: 处理后的图像数据，形状为 (C, H, W)。
       - label_sbbox: 小尺寸的标签数组，形状为 (H, W, A, 6 + C)，其中 A 为每个位置的锚点数量，C 为类别数。
       - label_mbbox: 中等尺寸的标签数组，形状为 (H, W, A, 6 + C)。
       - label_lbbox: 大尺寸的标签数组，形状为 (H, W, A, 6 + C)。
       - sbboxes: 小尺寸目标的边界框坐标，形状为 (M, 4)，其中 M 为小尺寸目标的数量。
       - mbboxes: 中等尺寸目标的边界框坐标，形状为 (M, 4)。
       - lbboxes: 大尺寸目标的边界框坐标，形状为 (M, 4)。
    """
    def __getitem__(self, item):
        # 确保索引在合理范围内
        # self 指的是数据集类的实例，而 len(self) 则是调用数据集实例的 __len__ 方法，用于获取数据集的长度或样本数量。
        assert item <= len(self), "index range error"

        # 解析原始图像和边界框注释
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
        
        # 随机选择另一个样本进行 Mixup 操作
        item_mix = random.randint(0, len(self.__annotations) - 1)  # random 模块的 randint 函数，用于生成一个指定范围内的随机整数。
        img_mix, bboxes_mix = self.__parse_annotation(
            self.__annotations[item_mix]
        )
        img_mix = img_mix.transpose(2, 0, 1)  # HWC->CHW

        # 使用 Mixup 数据增强
        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix
        
        # 创建标签
        (
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return (
            img,
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        )

    def __load_annotations(self, anno_type):
        # 加载指定类型的注释数据
        assert anno_type in [
            "train",
            "test",
        ], "You must choice one of the 'train' or 'test' for anno_type parameter"
        # 拼接注释文件路径
        anno_path = os.path.join(cfg.DATA_PATH, anno_type + "_annotation.txt")
        # 打开注释文件， 过滤掉空行
        with open(anno_path, "r") as f:
            # filter:内置函数（此处就是lambda函数），根据提供的函数对可迭代对象进行过滤。
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        # 确保注释列表不为空
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        # 返回注释列表
        return annotations
    """
    def __parse_annotation(self, annotation):函数逻辑
           2. 输入：
               - `annotation`：包含图像路径、边界框坐标和类别的注释信息，
               -  格式为 [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]。
           3. 功能：
              - 从注释信息中解析出图像路径和边界框信息。
              - 使用OpenCV读取图像，确保图像存在。
              - 将边界框信息转换为NumPy数组。
              - 应用一系列数据增强操作：
                - 随机水平翻转。
                - 随机裁剪。
                - 随机仿射变换。
                - 调整大小至指定的图像尺寸。
              - 返回增强后的图像和边界框。
           4. 输出：
              返回经过数据增强处理后的图像和边界框。这样的输出可用于提升模型对多样性数据的学习能力，提高模型的泛化性能。
    """
    def __parse_annotation(self, annotation):
        # 将注释字符串按空格分割
        anno = annotation.strip().split(" ")
        
        # 获取图像路径
        img_path = anno[0]  # 第一个位置存放的是image_path
        # OpenCV读取图像，读取的图像格式为[h*w*c], 通道顺序为BGR
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        # 检查图像是否成功读取
        assert img is not None, "File Not Found " + img_path
        
        # 将注释中的坐标信息转化为numpy数组
        bboxes = np.array(
            [list(map(float, box.split(","))) for box in anno[1:]]
        )  # map(float, ...) 则将列表中的每个元素都转换为浮点数。
        
        # 数据增强：水平翻转
        img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes), img_path)
        # 数据增强：随机裁剪
        img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        # 数据增强：随机放射变换
        img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        # 数据增强：调整大小
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(
            np.copy(img), np.copy(bboxes)
        )

        return img, bboxes
    
    
    """
    def __creat_label(self, bboxes):
    1. 总结：
       - def __creat_label(self, bboxes):函数用于为目标检测模型生成训练标签和边界框坐标。
       - 通过处理输入的边界框信息，计算它们与模型锚点的匹配关系，并生成相应的标签，最终返回用于训练的标签数组和边界框坐标。
    2. 输入：
       - bboxes: 输入的边界框信息，形状为 (N, 6)，其中 N 为边界框的数量。
       - 每行包括目标的图像路径、坐标信息（xmin, ymin, xmax, ymax）、类别和混合参数。
    3. 功能：
       - 根据输入的边界框信息，计算它们与模型锚点的匹配关系。
       - 生成训练标签数组，其中包括目标存在的概率、边界框坐标、类别信息等。
       - 处理同一目标可能匹配到多个锚点的情况，以及多个目标可能匹配到同一锚点的情况。
       - 返回生成的标签数组和相应的边界框坐标。
    4. 输出：
       - label_sbbox: 小尺寸的标签数组，形状为 (H, W, A, 6 + C)。
       - label_mbbox: 中等尺寸的标签数组，形状为 (H, W, A, 6 + C)。
       - label_lbbox: 大尺寸的标签数组，形状为 (H, W, A, 6 + C)。
       - 其中 H、W 分别为模型输出的特征图的高和宽，A 为每个位置的锚点数量，C 为类别数。
       - sbboxes: 小尺寸目标的边界框坐标，形状为 (M, 4)，其中 M 为小尺寸目标的数量。
       - mbboxes: 中等尺寸目标的边界框坐标，形状为 (M, 4)。
       - lbboxes: 大尺寸目标的边界框坐标，形状为 (M, 4)。
    """
    def __creat_label(self, bboxes):
        # 从配置文件中获取锚框和步长信息
        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        # 初始化标签
        label = [
            np.zeros(
                (
                    int(train_output_size[i]),
                    int(train_output_size[i]),
                    anchors_per_scale,
                    6 + self.num_classes,
                )
            )  # np.zeros(shape)：创建指定维度的全零数组
            for i in range(3)
        ]
        for i in range(3):
            label[i][..., 5] = 1.0
        
        # 初始化存储坐标信息的数组
        bboxes_xywh = [
            np.zeros((150, 4)) for _ in range(3)
        ]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )  # 非原地操作，创建看了新的numpy数组
            # print("bbox_xywh: ", bbox_xywh)
            
            # 处理坐标超过图像边界的情况
            for j in range(len(bbox_xywh)):
                if int(bbox_xywh[j]) >= self.img_size:  # self.img_size：416
                    differ = bbox_xywh[j] - float(self.img_size) + 1.
                    bbox_xywh[j] -= differ
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            )

            # 计算Iou
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                
                # 设置锚框中心坐标
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )  # 0.5 for compensation
                
                # 设置锚框的宽度和高度
                anchors_xywh[:, 2:4] = anchors[i]
                
                # 计算当前尺度下边界框与锚框的交并比
                iou_scale = tools.iou_xywh_numpy(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                # 将Iou保存到列表中
                iou.append(iou_scale)
                # 创建Iou掩码， 用于判断当前边界框是否与某个锚框有较高的重叠
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    # 如果存在与某个锚框有较高交并比的边界框
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    # 将边界框信息填充到对应的位置
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # 边界框坐标
                    label[i][yind, xind, iou_mask, 4:5] = 1.0  # 置信度
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix  # Mixup 系数
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth  # 类别信息

                    # 记录边界框的索引
                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大  %：取余操作
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    
                    # 标记存在正样本
                    exist_positive = True

            if not exist_positive:
                # 如果不存在正样本，则选择具有较高交并比的锚框作为最佳预测
                
                # 找到具有较高交并比的锚框的索引，将一维数组展平后取最大值的索引。
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                # 找到具有较高交并比的锚框的索引，将一维数组展平后取最大值的索引。
                best_detect = int(best_anchor_ind / anchors_per_scale)
                # 计算最佳预测所在尺度内的具体锚框。
                best_anchor = int(best_anchor_ind % anchors_per_scale)
                # 计算最佳预测的中心坐标在标签中的位置。
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)
                
                # 将边界框信息填充到标签中，包括边界框坐标、置信度、Mixup 系数和类别信息。
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh  # 边界框坐标
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0  # 置信度
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix  # Mixup系数
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth  # 类别信息
                
                # 记录边界框在数组中的索引，考虑到 BUG : 150 为一个先验值，内存消耗大。
                bbox_ind = int(bbox_count[best_detect] % 150)
                # 录边界框的信息。
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                # 更新边界框计数，以确保每个边界框都有唯一的索引。
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    
    # 创建VOC数据集
    voc_dataset = Build_Dataset(anno_file_type="train", img_size=448)
    # 创建数据加载器，用于批量加载数据
    dataloader = DataLoader(
        voc_dataset, shuffle=True, batch_size=1, num_workers=0
    )
    
    # 遍历数据加载器中的每个批次
    for i, (
        img,
        label_sbbox,
        label_mbbox,
        label_lbbox,
        sbboxes,
        mbboxes,
        lbboxes,
    ) in enumerate(dataloader):
        # 打印第一个批次的信息
        if i == 0:
            print(img.shape)  # 输出图像形状
            print(label_sbbox.shape)  # 输出小尺度边界框标签的形状
            print(label_mbbox.shape)  # 输出中尺度边界框标签的形状
            print(label_lbbox.shape)  # 输出大尺度边界框标签的形状
            print(sbboxes.shape)  # 输出小尺度边界框的形状
            print(mbboxes.shape)  # 输出中尺度边界框的形状
            print(lbboxes.shape)  # 输出大尺度边界框的形状

            # 如果图像批次大小为1
            if img.shape[0] == 1:
                # 将标签信息合并成一个数组
                labels = np.concatenate(
                    [
                        label_sbbox.reshape(-1, 26),
                        label_mbbox.reshape(-1, 26),
                        label_lbbox.reshape(-1, 26),
                    ],
                    axis=0,
                )
                # 根据置信度进行过滤
                labels_mask = labels[..., 4] > 0
                labels = np.concatenate(
                    [
                        labels[labels_mask][..., :4],
                        np.argmax(
                            labels[labels_mask][..., 6:], axis=-1
                        ).reshape(-1, 1),
                    ],
                    axis=-1,
                )

                print(labels.shape)  # 输出过滤后的标签信息的形状
                tools.plot_box(labels, img, id=1)  # 绘制边界框
