import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random  # 生成伪随机数的模块
import argparse  # 用于解析命令行参数的模块
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov4_config as cfg  # 导入配置文件
from utils import cosine_lr_scheduler
from utils.log import Logger
from apex import amp  # NVIDIA Apex 库中的一部分，用于实现混合精度训练。
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator


def detection_collate(batch):  # 将单个数据整合成一个batch_size
    """当然，让我们使用中文解释这段代码：
    1. **输入：**
       - `batch`：一个样本的批次。每个样本都是一个元组，其中第一个元素（`sample[0]`）是图像张量，第二个元素（`sample[1]`）是相应的目标信息。
    2. **功能：**
       - `detection_collate` 是在目标检测任务中常用的整理函数，特别是在 PyTorch 的 DataLoader 中使用。它将输入批次整理成适合进一步处理的格式。
       - 从每个样本中分离图像和目标。
    3. **输出：**
       - 返回一个包含两个元素的元组：
          - 一个张量（`torch.stack(imgs, 0)`），表示批次中堆叠的图像。
          - 一个列表（`targets`），包含批次中每个图像的目标信息。
    总的来说，这个函数接收一个样本批次，从每个样本中提取图像和目标信息，将图像堆叠成张量，并返回一个包含堆叠图像和目标列表的元组。这是目标检测任务中批次预处理的常见步骤。"""
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


class Trainer(object):
    """
        **总结：**
    该代码定义了一个名为`Trainer`的类，用于训练目标检测模型（YOLOv4或MobileNet-YOLO）。它支持多尺度训练，可以从断点继续训练，同时记录并保存训练过程中的关键信息，如损失、学习率、验证 mAP 等。
    
    **输入:**
    - `weight_path`: 权重文件路径，用于加载预训练权重或继续训练。
    - `resume`: 是否从断点继续训练。
    - `gpu_id`: GPU 设备 ID。
    - `accumulate`: 梯度累积步数。
    - `fp_16`: 是否使用混合精度训练。
    
    **实现的功能:**
    1. **初始化 (`__init__`):**
       - 设置随机种子、设备、起始训练周期、最佳 mAP、梯度累积步数等。
       - 创建训练数据集和数据加载器。
       - 构建 YOLOv4 模型、优化器、损失函数和学习率调度器。
       - 如果从断点恢复训练，加载之前保存的权重。
    
    2. **加载断点权重 (`__load_resume_weights`):**
       - 从最近保存的权重中加载模型状态、优化器状态、最佳 mAP 等信息。
    
    3. **保存模型权重 (`__save_model_weights`):**
       - 根据当前的 mAP 值保存模型权重。
       - 如果当前 mAP 优于历史最佳 mAP，则保存到 "best.pt" 文件中。
       - 每隔 10 个周期保存一次权重，并保存到 "backup_epoch%g.pt" 文件中。
    
    4. **训练 (`train`):**
       - 针对每个训练周期进行训练。
       - 遍历训练数据加载器，更新模型权重。
       - 使用学习率调度器动态调整学习率。
       - 打印训练信息，包括损失、学习率等。
       - 支持混合精度训练。
       - 在一定周期内，进行模型验证，计算 mAP。
       - 根据验证结果保存模型权重。
    
    **输出:**
    - 通过日志输出训练过程中的信息，包括训练损失、学习率、验证 mAP 等。
    """
    def __init__(self, weight_path=None,
                 resume=False,
                 gpu_id=0,
                 accumulate=1,
                 fp_16=False):
        init_seeds(0)  # 初始化随机种子
        self.fp_16 = fp_16  #是否使用混合精度训练
        self.device = gpu.select_device(gpu_id)  # 选择GPU设备
        self.start_epoch = 0  # 训练起始周期
        self.best_mAP = 0.0  # 最佳验证mAP
        self.accumulate = accumulate  # 梯度累积步数（每个batch_size作为一个step）
        self.weight_path = weight_path  # 模型权重保存路径
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]  # 是否进行多尺度训练(cfg是yolov4_config.py的别名)
        self.showatt = cfg.TRAIN["showatt"]  # 是否显示注意力图
        if self.multi_scale_train:
            print("Using multi scales training")
        else:
            print("train img size is {}".format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.train_dataset = data.Build_Dataset(
            anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )
        self.epochs = (
            cfg.TRAIN["YOLO_EPOCHS"]
            if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
            else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        )
        
        # 评估周期
        self.eval_epoch = (
            30 if cfg.MODEL_TYPE["TYPE"] == "YOLOv4" else 50
        )
        
        # 创建训练数据加载器
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True,
            pin_memory=True,  # 是否将数据加载到CUDA异步内存中（适用于GPU加速）
        )
        
        # 创建yolov4模型示例
        self.yolov4 = Build_Model(weight_path=weight_path, resume=resume, showatt=self.showatt).to(
            self.device
        )
        
        # 使用SGD优化器
        self.optimizer = optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

        # 损失函数
        self.criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )

        # 学习率调度器，使用余弦退火调度
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )
        
        # 如果是从断点继续训练，则加载已经保存的权重
        if resume:
            self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):
        # 构建最后一个保存的权重路径
        last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
        # 使用torch加载保存的权重文件， map_location=self.device指定模型当前所在的设备
        chkpt = torch.load(last_weight, map_location=self.device)
        # 将加载的权重应用于模型
        self.yolov4.load_state_dict(chkpt["model"])

        # 更新起始权重
        self.start_epoch = chkpt["epoch"] + 1
        # 如果优化器的的状态信息存在， 也加载优化器的状态
        if chkpt["optimizer"] is not None:
            self.optimizer.load_state_dict(chkpt["optimizer"])
            # 更新最佳mAP
            self.best_mAP = chkpt["best_mAP"]
        del chkpt  # 释放加载的权重

    def __save_model_weights(self, epoch, mAP):
        # 如果当前mAP优于最佳mAP,则更新最佳mAP
        if mAP > self.best_mAP:
            self.best_mAP = mAP
            
        # 构建最佳权重和最后一个权重的保存路径
        best_weight = os.path.join(
            os.path.split(self.weight_path)[0], "best.pt"
        )
        last_weight = os.path.join(
            os.path.split(self.weight_path)[0], "last.pt"
        )
        
        # 构建保存的checkpoint字典：用于保存模型的训练状态
        chkpt = {
            "epoch": epoch,  # 保存当前的训练周期（epoch）
            "best_mAP": self.best_mAP,  # 保存当前的最佳验证mAP
            "model": self.yolov4.state_dict(),  # 保存模型的状态字典
            "optimizer": self.optimizer.state_dict(),  # 保存优化器的状态字典，包含了优化器的状态信息，学习率和梯度等
        }
        
        # 保存最后一个权重
        torch.save(chkpt, last_weight)

        # 如果当前mAP是最佳mAP,则额外保存最佳权重
        if self.best_mAP == mAP:
            torch.save(chkpt["model"], best_weight)

        # 每十个周期保存一个权重
        if epoch > 0 and epoch % 10 == 0:
            torch.save(
                chkpt,
                os.path.join(
                    os.path.split(self.weight_path)[0],
                    "backup_epoch%g.pt" % epoch,
                ),
            )
            
        # 释放保存的checkpoint
        del chkpt

    def train(self):
        # 设置全局变量
        global writer
        # 打印训练相关的配置信息
        logger.info(
            "Training start,img size is: {:d},batchsize is: {:d},work number is {:d}".format(
                cfg.TRAIN["TRAIN_IMG_SIZE"],
                cfg.TRAIN["BATCH_SIZE"],
                cfg.TRAIN["NUMBER_WORKERS"],
            )
        )
        logger.info(self.yolov4)
        logger.info(
            "Train datasets number is : {}".format(len(self.train_dataset))
        )

        def is_valid_number(x):  # 判断x是否是有效的数字
            return not (math.isnan(x) or math.isinf(x) or x > 1e4)
        
        # 初始化混合精度训练
        if self.fp_16:
            self.yolov4, self.optimizer = amp.initialize(
                self.yolov4, self.optimizer, opt_level="O1", verbosity=0
            )
            
        # 训练开始的提示信息
        logger.info("        =======  start  training   ======     ")
        
        # 遍历训练周期
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()  # 记录当前时间，方便计算每个周期的训练时间
            self.yolov4.train()  # 将模型设置为训练状态

            mloss = torch.zeros(4)  # 记录多个损失项的平均值
            logger.info("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            
            # 遍历训练数据
            for i, (
                imgs,
                label_sbbox,
                label_mbbox,
                label_lbbox,
                sbboxes,
                mbboxes,
                lbboxes,
            ) in enumerate(self.train_dataloader):
                
                # 调整学习率
                self.scheduler.step(
                    len(self.train_dataloader)
                    / (cfg.TRAIN["BATCH_SIZE"])
                    * epoch
                    + i
                )
                
                # 将数据转到当前的设备上
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)
                
                # 将图像输入到模型进行预测
                p, p_d = self.yolov4(imgs)
                
                # 计算损失
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                    p,
                    p_d,
                    label_sbbox,
                    label_mbbox,
                    label_lbbox,
                    sbboxes,
                    mbboxes,
                    lbboxes,
                )
                
                # 如果损失值是有效的数值， 则进行反向传播
                if is_valid_number(loss.item()):
                    if self.fp_16:  # 检查是否启用了混合精度训练
                        # 如果启用了混合精度训练，采用amp.scale_loss函数对损失进行缩放
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()  # 基于混合精度训练进行反向传播
                    else:
                        loss.backward()  # 不采用混合进度进行训练的反向传播
                # Accumulate gradient for x batches before optimizing
                
                # 优化之前进行梯度累积
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # 更新运行时的损失均值
                loss_items = torch.tensor(
                    [loss_ciou, loss_conf, loss_cls, loss]
                )
                mloss = (mloss * i + loss_items) / (i + 1)

                # 打印每批次的结果
                if i % 10 == 0:

                    logger.info(
                        "  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                            epoch,
                            self.epochs,
                            i,
                            len(self.train_dataloader) - 1,
                            self.train_dataset.img_size,
                            mloss[3],
                            mloss[0],
                            mloss[1],
                            mloss[2],
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )
                    
                    # 将损失写入到Tensorboard
                    writer.add_scalar(
                        "loss_ciou",
                        mloss[0],  # 记录Ciou损失
                        len(self.train_dataloader)
                        * epoch
                        + i,
                    )
                    writer.add_scalar(
                        "loss_conf",
                        mloss[1],  # 记录置信度损失
                        len(self.train_dataloader)
                        * epoch
                        + i,
                    )
                    writer.add_scalar(
                        "loss_cls",
                        mloss[2],  # 记录分类损失
                        len(self.train_dataloader)
                        * epoch
                        + i,
                    )
                    writer.add_scalar(
                        "train_loss",
                        mloss[3],  # 记录总体训练损失， 包括ciou、置信度和分类损失
                        len(self.train_dataloader)
                        * epoch
                        + i,
                    )
                
                # multi-sclae training (320-608 pixels) every 10 batches
                # 多尺度训练（每10个批次调整一次图像大小）
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = (
                        random.choice(range(10, 20)) * 32
                    )  # 随机调整图像尺寸320---640,必须是32的倍数

            # 判断数据类型是否为“VOC”或者"Customer"
            if (
                cfg.TRAIN["DATA_TYPE"] == "VOC"
                or cfg.TRAIN["DATA_TYPE"] == "Customer"
            ):
                # 初始化mAP变量为0.0
                mAP = 0.0
                
                # 判断是否达到验证周期，达到了则进行下面操作
                if epoch >= self.eval_epoch:
                    # 打印相关日志信息
                    logger.info(
                        "===== Validate =====".format(epoch, self.epochs)
                    )
                    logger.info("val img size is {}".format(cfg.VAL["TEST_IMG_SIZE"]))
                    
                    # 不进行梯度计算情况下，使用Evaluator类进行VOC数据集的验证
                    with torch.no_grad():
                        # 调用Evaluator类中的APs_voc方法， 获取每个class的平均精度 （APs）
                        APs, inference_time = Evaluator(
                            self.yolov4, showatt=self.showatt
                        ).APs_voc()
                        
                        # 遍历每个类别的APs,打印和累加到总mAP
                        for i in APs:
                            logger.info("{} --> mAP : {}".format(i, APs[i]))
                            mAP += APs[i]
                        # 计算平均mAP
                        mAP = mAP / self.train_dataset.num_classes
                        logger.info("mAP : {}".format(mAP))
                        logger.info(
                            "inference time: {:.2f} ms".format(inference_time)
                        )
                        
                        # 将mAP写入TensorBoard
                        writer.add_scalar("mAP", mAP, epoch)
                        
                        # 保存的模型权重（如果当前的mAP更好）
                        self.__save_model_weights(epoch, mAP)
                        # 打印测试mAP
                        logger.info("save weights done")
                    logger.info("  ===test mAP:{:.3f}".format(mAP))
            
            # 如果当前数据集类型是COCO且当前epoch大于等于0
            elif epoch >= 0 and cfg.TRAIN["DATA_TYPE"] == "COCO":
                # 创建COCOAPIEvaluator实例，用于评估模型在COCO数据集上的性能
                evaluator = COCOAPIEvaluator(
                    model_type="YOLOv4",
                    data_dir=cfg.DATA_PATH,
                    img_size=cfg.VAL["TEST_IMG_SIZE"],
                    confthre=0.08,
                    nmsthre=cfg.VAL["NMS_THRESH"],
                )
                
                # 进行模型评估，获取COCO指标（ap50_95和ap50）
                ap50_95, ap50 = evaluator.evaluate(self.yolov4)
                logger.info("ap50_95:{}|ap50:{}".format(ap50_95, ap50))
                
                # 将COCO指标写入TensorBoard
                writer.add_scalar("val/COCOAP50", ap50, epoch)
                writer.add_scalar("val/COCOAP50_95", ap50_95, epoch)
                
                # 保存模型权重（如果当前ap50更好）
                self.__save_model_weights(epoch, ap50)
                print("save weights done")
            end = time.time()  # 记录评估结束时间
            logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info(
            "=====Training Finished.   best_test_mAP:{:.3f}%====".format(
                self.best_mAP
            )
        )


"""
1. 入参：
   - `weight_path`（数据类型：str，默认值："weight/mobilenetv2.pth"）：模型权重文件的路径
   - `resume`（数据类型：bool，默认值：False）：是否恢复训练的标志
   - `gpu_id`（数据类型：int，默认值：-1）：使用 GPU 的设备 ID，-1 表示使用 CPU
   - `log_path`（数据类型：str，默认值："log/"）：日志文件路径
   - `accumulate`（数据类型：int，默认值：2）：优化之前累积的批次数
   - `fp_16`（数据类型：bool，默认值：False）：是否使用 fp16 精度

2. 函数功能：
   - 当脚本独立运行时，通过命令行参数配置训练参数，并启动训练过程

3. 返回值：无

4. 总结：
   该脚本用于配置训练参数，创建日志记录器和事件记录器，并通过Trainer类启动训练过程。
"""
if __name__ == "__main__":
    global logger, writer
    # 解析命令行参数
    parser = argparse.ArgumentParser()  # argparse.ArgumentParser() 创建了一个 ArgumentParser 对象，用于解析命令行参数。
    
    # 定义程序需要接受的命令行参数（--weight_path, --resume， --gpu_id， --log_path， --accumulate， --fp_16）
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/mobilenetv2.pth",
        help="weight file path",
    )  # weight/darknet53_448.weights
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training flag",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(0) or CPU(-1)",
    )
    parser.add_argument("--log_path", type=str, default="log/", help="log path")
    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
        help="batches to accumulate before optimizing",
    )
    parser.add_argument(
        "--fp_16",
        type=bool,
        default=False,
        help="whither to use fp16 precision",
    )
    
    
    opt = parser.parse_args()  # parser.parse_args() 方法来解析命令行参数
    
    # 创建事件记录器
    writer = SummaryWriter(logdir=opt.log_path + "/event")
    # 创建日志记录器
    logger = Logger(
        log_file_name=opt.log_path + "/log.txt",
        log_level=logging.DEBUG,
        logger_name="YOLOv4",
    ).get_log()

    # 启动训练过程
    Trainer(
        weight_path=opt.weight_path,
        resume=opt.resume,
        gpu_id=opt.gpu_id,
        accumulate=opt.accumulate,
        fp_16=opt.fp_16,
    ).train()
