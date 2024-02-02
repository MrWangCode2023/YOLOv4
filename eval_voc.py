import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger


# 定义评估器类
class Evaluation(object):
    def __init__(
        self,
        gpu_id=0,
        weight_path=None,
        visiual=None,
        eval=False,
        mode=None
    ):
        # 从配置文件中获取类别数量、置信度阈值和非极大值抑制阈值
        self.__num_class = cfg.VOC_DATA["NUM"]
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        # 选择GPU设备
        self.__device = gpu.select_device(gpu_id)
        # 是否展示注意力图
        self.__showatt = cfg.TRAIN["showatt"]
        # 可视化路径
        self.__visiual = visiual
        # 模式，用于选择进行验证（val）还是检测（det）
        self.__mode = mode
        # 类别列表
        self.__classes = cfg.VOC_DATA["CLASSES"]
        # 构建YOLOv4模型，并移到所选择的设备上
        self.__model = Build_Model(showatt=self.__showatt).to(self.__device)
        # 加载预训练权重
        self.__load_model_weights(weight_path)
        # 创建Evaluator对象，传递YOLOv4模型和是否展示注意力图的信息
        self.__evalter = Evaluator(self.__model, showatt=self.__showatt)
    
    # 私有方法，加载预训练权重
    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))
        # 组合权重文件路径
        weight = os.path.join(weight_path)
        # 使用torch加载权重
        chkpt = torch.load(weight, map_location=self.__device)
        # 将权重加载到模型
        self.__model.load_state_dict(chkpt["model"])
        print("loading weight file is done")
        # 释放加载的权重文件
        del chkpt
    
    # 验证函数，用于评估模型性能
    def val(self):
        global logger
        logger.info("***********Start Evaluation****************")
        start = time.time()
        mAP = 0
        
        # 禁用梯度计算，因为在验证阶段不需要进行反向传播
        with torch.no_grad():
                # 使用Evaluator类计算VOC数据集的AP和推理时间
                APs, inference_time = Evaluator(
                    self.__model, showatt=False
                ).APs_voc()  # 类方法调用
                
                # 遍历每个类别的AP，并打印结果
                for i in APs:
                    logger.info("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                logger.info("mAP:{}".format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
        end = time.time()
        logger.info("  ===val cost time:{:.4f}s".format(end - start))
    
    # 目标检测函数，用于在指定路径下检测图像并保存结果
    def detection(self):
        global logger
        
        # 检查是否提供了可视化图像路径
        if self.__visiual:
            # 获取图像文件列表
            imgs = os.listdir(self.__visiual)
            logger.info("***********Start Detection****************")  # 打印日志
            
            # 遍历每张图像进行检测
            for v in imgs:
                path = os.path.join(self.__visiual, v)
                logger.info("val images : {}".format(path))
                
                # 读取图像
                img = cv2.imread(path)
                assert img is not None
                
                # 获取模型对图像的预测结果
                bboxes_prd = self.__evalter.get_bbox(img, v, mode=self.__mode)
                
                # 如果有检测结果
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]  # 检测框坐标
                    class_inds = bboxes_prd[..., 5].astype(np.int32)  # 类别索引
                    scores = bboxes_prd[..., 4]  # 检测置信度

                    # 可视化检测结果并保存图像
                    visualize_boxes(
                        image=img,
                        boxes=boxes,
                        labels=class_inds,
                        probs=scores,
                        class_labels=self.__classes,
                    )
                    
                    # 保存图像到指定路径
                    path = os.path.join(
                        cfg.PROJECT_PATH, "detection_result/{}".format(v)
                    )
                    cv2.imwrite(path, img)
                    logger.info("saved images : {}".format(path))


if __name__ == "__main__":
    global logger
    
    # 创建命令行对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--log_val_path", type=str, default="log_val", help="val log file path"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)",
    )
    parser.add_argument(
        "--visiual",
        type=str,
        default="VOCtest-2007/VOC2007/JPEGImages",
        help="det data path or None",
    )
    parser.add_argument("--mode", type=str, default="val", help="val or det")
    # 解析命令行参数
    opt = parser.parse_args()
    
    # 如果日志路径不存在，则创建该路径
    if not os.path.exists(opt.log_val_path):
        os.mkdir(opt.log_val_path)
    # 初始化日志记录器
    logger = Logger(
        log_file_name=opt.log_val_path + "/log_voc_val.txt",
        log_level=logging.DEBUG,
        logger_name="YOLOv4",
    ).get_log()

    # 根据模式选择进行验证或检测
    if opt.mode == "val":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            mode=opt.mode
        ).val()  # 验证模式
    else:
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            mode=opt.mode
        ).detection()  # 推理模式
