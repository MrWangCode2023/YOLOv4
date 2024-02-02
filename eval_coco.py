import utils.gpu as gpu
from model.build_model import Build_Model
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger
import cv2
from eval.cocoapi_evaluator import COCOAPIEvaluator


class Evaluation(object):
    # 初始化方法，用于设置评估对象的各种属性和参数
    def __init__(self, gpu_id=0, weight_path=None, visiual=None, heatmap=False):
        self.__num_class = cfg.COCO_DATA["NUM"]  # COCO数据集的类别数量
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]  # 检测结果置信度阈值
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]  # 非极大值抑制的阈值
        self.__device = gpu.select_device(gpu_id)  # 所选择的GPU设备
        self.__multi_scale_val = cfg.VAL["MULTI_SCALE_VAL"]  # 是否使用多尺度验证
        self.__flip_val = cfg.VAL["FLIP_VAL"]  # 是否进行水平翻转验证

        self.__visiual = visiual  # 可视化数据的路径
        self.__eval = eval
        self.__classes = cfg.COCO_DATA["CLASSES"]  # COCO数据集的类别列表

        self.__model = Build_Model().to(self.__device)  # 构建目标检测模型并将其移动到GPU上

        self.__load_model_weights(weight_path)  # 加载模型权重

        self.__evalter = Evaluator(self.__model, showatt=heatmap)  # 创建评估器对象，传入模型和是否显示注意力热图的参数

    def __load_model_weights(self, weight_path):
        # 打印加载权重文件的信息
        print("loading weight file from : {}".format(weight_path))
       
        # 构建权重文件路径
        weight = os.path.join(weight_path)
        # 加载权重文件
        chkpt = torch.load(weight, map_location=self.__device)
        
        # 将模型的状态字典更新为加载的权重
        self.__model.load_state_dict(chkpt)
        # 打印加载完成的信息
        print("loading weight file is done")
        
        # 释放内存
        del chkpt

    def reset(self):
        # 重置检测结果路径
        path1 = os.path.join(cfg.DETECTION_PATH, "detection_result/")
        path2 = os.path.join(cfg.DETECTION_PATH, "ShowAtt/")
        
        # 删除detection_result目录下的所有文件和子目录中的文件
        for i in os.listdir(path1):
            path_file = os.path.join(path1, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)
                        
        # # 删除ShowAtt目录下的所有文件和子目录中的文件
        for i in os.listdir(path2):  # path2 = os.path.join(cfg.DETECTION_PATH, "ShowAtt/")
            path_file = os.path.join(path2, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)

    def study(self):
        # Parameter study
        y = []
        
        # 针对不同的置信度阈值进行评估
        for i in [0.08, 0.07, 0.06]:
            t = time.time()
            
            # 创建COCOAPI评估器
            evaluator = COCOAPIEvaluator(
                model_type="YOLOv3",
                data_dir=cfg.DATA_PATH,
                img_size=cfg.VAL["TEST_IMG_SIZE"],
                confthre=i,
                nmsthre=cfg.VAL["NMS_THRESH"],
            )
            
            # 进行评估
            _, r = evaluator.evaluate(self.__model)
            
            # 将结果记录到列表中
            y.append(
                str(i)
                + str("  ")
                + str(r)
                + str("  ")
                + str(
                    time.time() - t,
                )
            )
            
            # 保存结果到study.txt文件
            np.savetxt("study.txt", y, fmt="%s")  # y = np.loadtxt('study.txt')

    def val(self):
        global logger
        
        # 打印开始评估的信息
        logger.info("***********Start Evaluation****************")
        start = time.time()

        # 创建COCOAPI评估器
        evaluator = COCOAPIEvaluator(
            model_type="YOLOv4",
            data_dir=cfg.DATA_PATH,
            img_size=cfg.VAL["TEST_IMG_SIZE"],
            confthre=cfg.VAL["CONF_THRESH"],
            nmsthre=cfg.VAL["NMS_THRESH"],
        )
        
        # 进行评估
        ap50_95, ap50 = evaluator.evaluate(self.__model)
        
        # 打印评估结果
        logger.info("ap50_95:{}|ap50:{}".format(ap50_95, ap50))
        
        # 打印评估耗时
        end = time.time()
        logger.info("  ===val cost time:{:.4f}s".format(end - start))

    def Inference(self):
        global logger
        # 清除缓存
        self.reset()

        # 打印开始推理的信息
        logger.info("***********Start Inference****************")
        
        # 获取待推理的图像列表
        imgs = os.listdir(self.__visiual)
        logger.info("images path: {}".format(self.__visiual))
        
        # 设置推理结果保存路径
        path = os.path.join(cfg.DETECTION_PATH, "detection_result")
        logger.info("saved images at: {}".format(path))
        
        # 初始化推理时间列表
        inference_times = []
        
        # 遍历每张待推理的图像
        for v in imgs:
            start_time = time.time()
            path = os.path.join(self.__visiual, v)
            # 读取图像
            img = cv2.imread(path)
            assert img is not None

            # 获取推理结果
            bboxes_prd = self.__evalter.get_bbox(img, v)
            
            # 如果存在预测框，则可视化并保存结果图像
            if bboxes_prd.shape[0] != 0:
                # 提取预测框坐标、类别索引和置信度得分
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                # 可视化预测结果并保存图像
                visualize_boxes(
                    image=img,
                    boxes=boxes,
                    labels=class_inds,
                    probs=scores,
                    class_labels=self.__classes,
                )
                path = os.path.join(
                    cfg.DETECTION_PATH, "detection_result/{}".format(v)
                )
                cv2.imwrite(path, img)
                
            # 记录推理结束时间
            end_time = time.time()
            # 计算并记录本次推理的时间
            inference_times.append(end_time - start_time)
            
        # 计算所有推理时间的平均值和每秒推理的帧数
        inference_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / inference_time
        
        # 打印推理时间和FPS信息
        logging.info(
            "Inference_Time: {:.5f} s/image, FPS: {}".format(
                inference_time, fps
            )
        )


if __name__ == "__main__":
    # 初始化全局日志记录器
    global logger
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--log_val_path", type=str, default="log_val", help="weight file path"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)",
    )
    parser.add_argument(
        "--visiual", type=str, default="test_pic", help="val data path or None"
    )
    parser.add_argument(
        "--mode", type=str, default="val", help="val or det or study"
    )
    parser.add_argument(
        "--heatmap", type=str, default=False, help="whither show attention map"
    )
    # 解析命令行参数
    opt = parser.parse_args()
    
    # 初始化日志记录器
    logger = Logger(
        log_file_name=opt.log_val_path + "/log_coco_val.txt",  # 指定日志文件路径
        log_level=logging.DEBUG,  # 设置日志级别为DEBUG，记录详细信息
        logger_name="YOLOv4",  # 日志记录器的名称
    ).get_log()

    # 根据模式选择相应的操作
    if opt.mode == "val":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).val()  # 进行模型验证
    if opt.mode == "det":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).Inference()  # 进行模型推理
    else:
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            heatmap=opt.heatmap,
        ).study()  # 进行模型参数研究
