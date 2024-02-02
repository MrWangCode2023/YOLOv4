import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
from timeit import default_timer as timer
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger
from tensorboardX import SummaryWriter


class Detection(object):
    def __init__(
        self,
        gpu_id=0,
        weight_path=None,
        video_path=None,
        output_dir=None,
    ):
        # 读取VOC数据集中的类别数量
        self.__num_class = cfg.VOC_DATA["NUM"]
        
        # 获取验证时的置信度和非极大值抑制的阈值
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        
        # 选择使用的计算设备（GPU或CPU）
        self.__device = gpu.select_device(gpu_id)
        
        # 是否进行多尺度验证和翻转验证
        self.__multi_scale_val = cfg.VAL["MULTI_SCALE_VAL"]
        self.__flip_val = cfg.VAL["FLIP_VAL"]
        
        # 获取VOC数据集的类别标签
        self.__classes = cfg.VOC_DATA["CLASSES"]
        
        # 视频路径和输出目录
        self.__video_path = video_path
        self.__output_dir = output_dir
        
        # 创建YOLO模型实例并将其移到计算设备上
        self.__model = Build_Model().to(self.__device)
        
        # 载入预训练权重
        self.__load_model_weights(weight_path)
        
        # 创建评估器实例
        self.__evalter = Evaluator(self.__model, showatt=False)

    def __load_model_weights(self, weight_path):
        # 打印正在加载的权重文件路径
        print("loading weight file from : {}".format(weight_path))
        
        # 将权重文件路径与操作系统的路径结合
        weight = os.path.join(weight_path)
        
        # 使用PyTorch的torch.load函数加载权重文件，map_location用于指定加载到的设备
        chkpt = torch.load(weight, map_location=self.__device)
        
        # 将加载的权重加载到模型中
        self.__model.load_state_dict(chkpt)
        # 打印加载完成信息
        print("loading weight file is done")
        # 释放加载的权重文件，以释放内存
        del chkpt

    def Video_detection(self):
        import cv2
        
        # 打开视频文件
        vid = cv2.VideoCapture(self.__video_path)
        
        # 检查视频是否成功打开，否则抛出错误
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        
        # 获取视频的参数，包括FourCC编码、帧率和尺寸
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))  # FourCC 主要用于视频文件和流中，以指定视频编解码器的类型。
        video_fps = vid.get(cv2.CAP_PROP_FPS)  # 帧率
        video_size = (
            int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),  # 视频帧宽度
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),  # 视频帧高度
        )
        
        # 判断是否输出结果
        isOutput = True if self.__output_dir != "" else False
        if isOutput:
            print(
                "!!! TYPE:",
                type(self.__output_dir),
                type(video_FourCC),
                type(video_fps),
                type(video_size),
            )
            # 如果输出，创建VideoWrider对象
            out = cv2.VideoWriter(
                self.__output_dir, video_FourCC, video_fps, video_size
            )
        
        # 初始化一些变量，用于计算FPS
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        
        # 循环读取视频帧
        while True:
            # 读取一帧
            return_value, frame = vid.read()
            
            # 获取预测的边界框
            bboxes_prd = self.__evalter.get_bbox(frame)
            
            # 如果有检测结果
            if bboxes_prd.shape[0] != 0:
                # 获取边界框的坐标、类别和置信度
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                
                # 可视化边界框在图像上
                visualize_boxes(
                    image=frame,
                    boxes=boxes,
                    labels=class_inds,
                    probs=scores,
                    class_labels=self.__classes,
                )
            
            # 计算FPS
            curr_time = timer()  # 获取当前时间
            exec_time = curr_time - prev_time  # 计算从上一帧到当前帧的执行时间
            prev_time = curr_time  # 更新prev_time为当前时间，以备下一帧使用
            accum_time = accum_time + exec_time  # 将exec_time累加到总执行时间
            curr_fps = curr_fps + 1  # 帧率计数器自增1
            
            # 每秒更新一次FPS
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
                
            # 在图像上绘制FPS
            cv2.putText(
                frame,
                text=fps,
                org=(3, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=(255, 0, 0),
                thickness=2,
            )
            
            # 创建一个可调整大小的窗口并显示图像
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)
            
            # 如果有输出，将帧写入输出文件
            if isOutput:
                out.write(frame)
            # 检测按键，如果按下 'q' 键则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    global logger, writer
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default="E:\YOLOV4\weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--video_path", type=str, default="bag.avi", help="video file path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="output file path"
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
    parser.add_argument("--mode", type=str, default="det", help="val or det")
    opt = parser.parse_args()
    
    # 创建TensorBoardXd 的SummaryWriter
    writer = SummaryWriter(logdir=opt.log_val_path + "/event")
    # 创建日志记录器
    logger = Logger(
        log_file_name=opt.log_val_path + "/log_video_detection.txt",
        log_level=logging.DEBUG,
        logger_name="CIFAR",
    ).get_log()

    # 创建Detection实例并调用Video_detection方法
    Detection(
        gpu_id=opt.gpu_id,
        weight_path=opt.weight_path,
        video_path=opt.video_path,
        output_dir=opt.output_dir,
    ).Video_detection()
