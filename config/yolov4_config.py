"""这段代码是一个项目的配置文件，主要用于设置目标检测模型的训练和验证参数。让我按照你提供的顺序来总结、解释：
1. **总结：**
   - 项目路径和数据路径的定义。
   - 模型类型、卷积类型和注意力类型的设置。
   - 训练和验证阶段的参数配置，包括数据类型（VOC、COCO或自定义）、图像大小、数据增强、批处理大小等。
   - 数据集相关的配置，如类别数和类别名称。
   - 模型结构相关的配置，如锚点、步长等。
2. **输入（实现的功能）：**
   - 代码接受用户定义的模型类型、卷积类型、注意力类型等作为输入。
   - 用户可以指定训练数据集的类型（VOC、COCO或自定义）和相应的配置。
   - 用户可以调整训练和验证阶段的超参数，如图像大小、批处理大小等。
   - 用户可以定义自己的数据集，包括类别数和类别名称。
3. **输出的模式解释：**
   - 代码通过定义了训练和验证阶段的参数，包括图像大小、数据增强、批处理大小等。
   - 模型结构相关的配置，如锚点、步长等，也在代码中进行了定义。
   - 用户可以根据项目的需要，灵活调整这些参数，以满足不同的目标检测任务。
总体而言，这段代码是一个目标检测模型的配置文件，通过设置各种参数和超参数，用户可以根据具体的需求来训练和验证模型。"""
# coding=utf-8
# project
import os.path as osp

# 项目路径和数据路径设置
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_PATH = osp.join(PROJECT_PATH, 'data')

"""
模型类型设置（本项目包含基本yolov4s基本模型以及几个变体模型）
YOLO type:YOLOv4, Mobilenet-YOLOv4, CoordAttention-YOLOv4 or Mobilenetv3-YOLOv4
"""
MODEL_TYPE = {
    "TYPE": "CoordAttention-YOLOv4"
}

# 卷积类型设置
CONV_TYPE = {"TYPE": "DO_CONV"}  # conv type:DO_CONV or GENERAL

# 注意力类型设置
ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE

# 训练阶段参数设置
TRAIN = {
    "DATA_TYPE": "VOC",  # DATA_TYPE: VOC ,COCO or Customer
    "TRAIN_IMG_SIZE": 416,              # 训练图像大小
    "AUGMENT": True,                    # 是否进行数据增强
    "BATCH_SIZE": 1,                    # 批处理大小（相当于step步长）
    "MULTI_SCALE_TRAIN": True,          # 是否进行多尺度训练
    "IOU_THRESHOLD_LOSS": 0.5,          # 损失函数里的Iou阈值
    "YOLO_EPOCHS": 50,                  # YOLO模型的训练轮数
    "Mobilenet_YOLO_EPOCHS": 120,       # Mobilenet-YOLO模型的训练轮数
    "NUMBER_WORKERS": 0,                # 训练时的工作线程数
    "MOMENTUM": 0.9,                    # 动量参数
    "WEIGHT_DECAY": 0.0005,             # 权重衰减参数
    "LR_INIT": 1e-4,                    # 初始学习率
    "LR_END": 1e-6,                     # 最终学习率
    "WARMUP_EPOCHS": 2,  # or None      # 初始学习率逐渐升高的轮数，或者设置为None
    "showatt": False                    # 是否显示注意力图（attention map）
}


# 验证阶段参数设置
VAL = {
    "TEST_IMG_SIZE": 416,               # 测试图像大小
    "BATCH_SIZE": 1,                    # 批处理大小
    "NUMBER_WORKERS": 0,                # 验证时的工作线程数
    "CONF_THRESH": 0.005,               # 置信度阈值
    "NMS_THRESH": 0.45,                 # 非极大值抑制阈值
    "MULTI_SCALE_VAL": False,           # 是否进行多尺度验证
    "FLIP_VAL": False,                  # 是否进行水平翻转验证
    "Visual": False,                    # 是否可视化验证结果
    "showatt": False                    # 是否显示注意力图（注意力图展示了模型在处理输入时关注的特定区域或通道的权重分布。）
}

# 自定义数据集设置
Customer_DATA = {
    "NUM": 3,  # 数据集类别数
    "CLASSES": ["unknown", "person", "car"],  # 数据集类别名称
}

# VOC数据集设置
VOC_DATA = {
    "NUM": 20,
    "CLASSES": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
}

# COCO数据集设置
COCO_DATA = {
    "NUM": 80,
    "CLASSES": [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
}


# 模型参数设置
MODEL = {
    "ANCHORS": [
        [
            (1.25, 1.625),
            (2.0, 3.75),
            (4.125, 2.875),
        ],  # Anchors for small obj(12,16),(19,36),(40,28)
        [
            (1.875, 3.8125),
            (3.875, 2.8125),
            (3.6875, 7.4375),
        ],  # Anchors for medium obj(36,75),(76,55),(72,146)
        [
            (3.625, 2.8125),
            (4.875, 6.1875),
            (11.65625, 10.1875)
        ],  # Anchors for big obj(142,110),(192,243),(459,401)
    ],
    "STRIDES": [8, 16, 32],         # 特征图的步长
    "ANCHORS_PER_SCLAE": 3,         # 每个尺度的锚点数
}
