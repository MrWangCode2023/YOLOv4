import shutil
from eval import voc_eval
from utils.data_augment import *
from utils.tools import *
from tqdm import tqdm
from utils.visualize import *
from utils.heatmap import imshowAtt
import config.yolov4_config as cfg
import time
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool        # 线程池
from collections import defaultdict
current_milli_time = lambda: int(round(time.time() * 1000))


class Evaluator(object):
    def __init__(self, model=None, showatt=False):
        # 根据数据类型设置类别信息
        if cfg.TRAIN["DATA_TYPE"] == "VOC":
            self.classes = cfg.VOC_DATA["CLASSES"]
        elif cfg.TRAIN["DATA_TYPE"] == "COCO":
            self.classes = cfg.COCO_DATA["CLASSES"]
        else:
            self.classes = cfg.Customer_DATA["CLASSES"]
        
        # 设置预测结果保存路径
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, "pred_result")
        # 设置验证数据路径
        self.val_data_path = os.path.join(
            cfg.DATA_PATH, "VOCtest-2007", "VOCdevkit", "VOC2007"
        )
        
        # 设置置信度和NMS的阈值
        self.conf_thresh = cfg.VAL["CONF_THRESH"]
        self.nms_thresh = cfg.VAL["NMS_THRESH"]
        
        # 设置验证图像的尺寸
        self.val_shape = cfg.VAL["TEST_IMG_SIZE"]
        
        # 设置模型和设备
        self.model = model
        self.device = next(model.parameters()).device
        
        # 初始化可视化图像数量、多尺度测试、翻转测试、是否显示注意力图的标志
        self.visual_imgs = 0
        self.multi_scale_test = cfg.VAL["MULTI_SCALE_VAL"]
        self.flip_test = cfg.VAL["FLIP_VAL"]
        self.showatt = showatt
        
        # 初始化推理时间和最终结果
        self.inference_time = 0.0
        self.final_result = defaultdict(list)

    """
    def APs_voc(self):
    **方法讲解: 计算目标检测性能指标 (APs_voc)**
    1. **入参**:
       - `self`: 类的实例。
    2. **实现功能**:
       - **获取图像索引列表**:
         - **数据意义**: 用于获取测试集中每张图像的索引。
         - **数据结构**: 列表。
         - **数据类型**: 字符串。
         
       - **创建结果保存路径**:
         - **数据意义**: 用于保存每个类别的检测结果。
         - **数据结构**: 字符串路径。
         - **数据类型**: 无。
    
       - **并行处理图像**:
         - **数据意义**: 通过多线程池并行处理每张图像，提高处理效率。
         - **数据结构**: 列表。
         - **数据类型**: 无。
    
       - **保存结果**:
         - **数据意义**: 保存每个类别的检测结果。
         - **数据结构**: 字符串。
         - **数据类型**: 无。
    
       - **计算性能指标**:
         - **数据意义**: 用于评估目标检测性能。
         - **数据结构**: 字符串和浮点数。
         - **数据类型**: 元组。
    3. **返回值**:
       - **数据意义**: 包括性能指标 (如平均精度 APs) 和推理时间。
       - **数据结构**: 元组。
       - **数据类型**: 浮点数。
    4. **总结**:
       - 该方法实现了在VOC数据集上计算目标检测性能指标的功能，包括多线程处理图像、保存检测结果、计算性能指标等步骤。返回的结果包括性能指标和推理时间。
    """
    def APs_voc(self):
        # 读取测试集中的图像列表
        img_inds_file = os.path.join(
            self.val_data_path, "ImageSets", "Main", "test.txt"
        )
        
        # 打开图像索引文件，并进行
        with open(img_inds_file, "r") as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]  # 去除每一行的首尾空白字符

        # 如果存在先前的预测结果文件夹，则删除
        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)

        # 创建预测结果文件夹
        output_path = "./output/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        os.mkdir(self.pred_result_path)
        
        # 获取图像总数、CPU核心数和线程池
        imgs_count = len(img_inds)
        cpu_nums = multiprocessing.cpu_count()
        pool = ThreadPool(cpu_nums)
        
        # 使用tqdm显示处理进度
        with tqdm(total=imgs_count) as pbar:
            # 多线程处理每张图像
            for i, _ in enumerate(
                    pool.imap_unordered(
                        self.Single_APs_voc, img_inds
                    )
            ):
                pbar.update()
        
        # 将每个类别的预测结果写入对应的文件
        for class_name in self.final_result:
            with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                str_result = ''.join(self.final_result[class_name])
                f.write(str_result)
                
        # 计算平均精度（AP）和推理时间
        self.inference_time = 1.0 * self.inference_time / len(img_inds)  # 单张图像推理时间
        return self.__calc_APs(), self.inference_time
    """
    **方法讲解: 处理单张图像的目标检测结果 (Single_APs_voc)**
    1. **输入**:
       - `self`: 类的实例。
       - `img_ind`: 当前处理的图像索引。
    2. **算法流程**:
       - **构建图像路径**:
         1. 根据当前图像索引构建图像路径，路径为 "JPEGImages/img_ind.jpg"。
       - **获取预测框**:
         1. 使用 `get_bbox` 方法获取图像的目标检测结果，返回的预测框信息包括位置坐标、置信度、类别等。
       - **可视化处理**:
         1. 如果预测框的数量不为零，且可视化图像数量未达到上限 (100)，进行可视化处理。
         2. 将预测框的信息绘制在原图上，保存可视化结果。
       - **结果保存**:
         1. 遍历每个预测框，将框的信息格式化并保存到对应类别的结果列表中。
         2. 格式为 "img_ind score xmin ymin xmax ymax\n"。
    3. **输出**:
       - 无返回值。
       - 可视化结果保存在指定路径。
       - 结果信息保存到类的成员变量 `final_result` 中。
    """
    def Single_APs_voc(self, img_ind):
        # 构建图像文件路径
        img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind + '.jpg')
        
        # 读取图像
        img = cv2.imread(img_path)
        # 获取预测的边界框
        bboxes_prd = self.get_bbox(img, self.multi_scale_test, self.flip_test)
        # 如果预测的边界框不为空且可视化图像数量未达到上限（100）
        if bboxes_prd.shape[0] != 0  and self.visual_imgs < 100:
            # 获取预测的边界框坐标、类别、置信度
            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]
            
            # 可视化边界框，将结果保存到img文件中
            visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes) # 原地操作
            path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.visual_imgs))
            cv2.imwrite(path, img)

            # 更新已经可视化的图形数量
            self.visual_imgs += 1
            
            
        # 遍历所有预测的边界框
        for bbox in bboxes_prd:
            coor = np.array(bbox[:4], dtype=np.int32)  # 坐标
            score = bbox[4]  # 置信度
            class_ind = int(bbox[5])  # 类别索引

            class_name = self.classes[class_ind]  # 类别名称
            score = '%.4f' % score  # 格式化为字符串保留四位小数
            xmin, ymin, xmax, ymax = map(str, coor)  # 解包坐标
            result = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'  # 新的字符串result:[img_ind, score, xmin, ymin, xmax, ymax]

            # 将结果添加到最终结果字典中，按类保存
            self.final_result[class_name].append(result)
            """
            # {class_name: "img_ind
                            score
                            xmin
                            ymin
                            xmax
                            ymax"}"""

    """
    **方法讲解: 获取目标框 (get_bbox)**
    1. **入参**:
       - **数据形状**: 输入图像的形状，通常为 (Height, Width, Channels)。
       - **数据含义**: 输入图像用于目标检测。
       - **数据类型**: 多维数组，NumPy数组。
    2. **函数功能**:
       - 该函数用于获取图像中的目标框。
       - 如果启用多尺度测试 (`multi_test=True`)，则在不同输入尺寸下进行目标检测，支持检测不同大小的目标。
       - 如果启用水平翻转测试 (`flip_test=True`)，则对水平翻转后的图像进行目标检测，增加模型的鲁棒性。
       - 最终通过非极大值抑制（NMS）处理，剔除重叠较多的目标框，返回最终的目标框坐标和相关信息。
    3. **返回值**:
       - **数据形状**: 目标框的坐标和相关信息，形状为 (N, 6)，其中 N 为检测到的目标框数量，每个目标框包括坐标信息（xmin, ymin, xmax, ymax）、置信度、类别索引。
       - **数据含义**: 目标框的坐标和相关信息。
       - **数据类型**: 多维数组，NumPy数组。
    """
    def get_bbox(self, img, multi_test=False, flip_test=False, mode=None):
        # 如果进行多尺度测试
        if multi_test:
            test_input_sizes = range(320, 640, 96)  # 创建一个320到640的迭代器，步长为96
            
            bboxes_list = []
            # 对每个尺度进行测试
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(
                    self.__predict(img, test_input_size, valid_scale, mode)
                )
                
                # 如果进行水平翻转测试
                if flip_test:
                    bboxes_flip = self.__predict(
                        img[:, ::-1], test_input_size, valid_scale, mode
                    )  # 通过逐行逆序进行水平翻转
                    bboxes_flip[:, [0, 2]] = (
                        img.shape[1] - bboxes_flip[:, [2, 0]]
                    )
                    bboxes_list.append(bboxes_flip)
                    
            # 合并不同尺度的预测结果
            bboxes = np.row_stack(bboxes_list)
        else:
            # 单尺度测试
            bboxes = self.__predict(img, self.val_shape, (0, np.inf), mode)
        # 应用非极大值抑制（nms）
        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes


    """
    ### 函数解释：
    1. **入参：**
        - **img:** 输入的图像，形状为 `(H, W, C)`，表示图像的高、宽和通道数。
        - **test_shape:** 模型输入的图像尺寸，用于调整输入图像的大小。
        - **valid_scale:** 有效尺度范围，限制目标框的大小。
        - **mode:** 模型运行模式，这里使用了字符串类型，可能为 'det'（检测模式）或其他值。
    
    2. **函数功能：**
        - **预测模型输出：** 通过将输入图像传递给模型，获取模型的输出。
        - **处理预测结果：** 将模型输出转换为包含目标框信息的格式。
        - **可视化注意力图（可选）：** 如果配置中启用了 `showatt`（注意力图显示），则可视化注意力图。
    
    3. **返回值：**
        - 返回包含目标框信息的数组，每行表示一个目标框的坐标信息、置信度和类别索引。
        - 数据形状：`(N, 6)`，其中 `N` 为检测到的目标框数量。
        - 数据含义：
            - 目标框的坐标信息包括了 xmin, ymin, xmax, ymax。
            - 置信度表示目标框的置信度，即模型认为该目标框包含目标的概率。
            - 类别索引表示目标框所属的类别索引。
    
    4. **总结：**
        - 该函数通过将输入图像传递给模型，获取模型的输出，并将输出转换为目标框的格式。如果启用了注意力图显示，还会可视化注意力图。
    """
    def __predict(self, img, test_shape, valid_scale, mode):
        # 备份原图像
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        # 将图像转化为模型输入的张量格式
        img = self.__get_img_tensor(img, test_shape).to(self.device)
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 在不计算梯度的条件下进行推理
        with torch.no_grad():
            start_time = current_milli_time()
            
            # 如果需要显示注意力图， 则获取注意力图
            if self.showatt:
                _, p_d, atten = self.model(img)
            # 不获取注意力图
            else:
                _, p_d = self.model(img)
                
            # 统计推理时间
            self.inference_time += current_milli_time() - start_time
            
        # 将预测结果从张量的的格式转化为numpy数组
        pred_bbox = p_d.squeeze().cpu().numpy()
        
        # 转换预测结果的坐标
        bboxes = self.__convert_pred(
            pred_bbox, test_shape, (org_h, org_w), valid_scale
        )
        
        # 如果需要显示注意力图，并且在det模式下，则显示注意力图
        if self.showatt and len(img) and mode == 'det':
            self.__show_heatmap(atten, org_img)
        return bboxes

    def __show_heatmap(self, beta, img):
        # 显示注意力图
        imshowAtt(beta, img)

    def __get_img_tensor(self, img, test_shape):
        # 将图像缩放到指定的测试形状，保持原始框不变，转置图像维度
        img = Resize((test_shape, test_shape), correct_box=False)(
            img, None
        ).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()

    """
    1. 入参：
       - `pred_bbox`（形状：(N, 5 + num_classes)，数据类型：float32）：包含预测框的坐标、置信度和类别概率信息的数组
       - `test_input_size`（数据类型：int）：测试输入尺寸
       - `org_img_shape`（形状：(h, w)，数据类型：tuple 或 list）：原始图像的形状
       - `valid_scale`（数据类型：tuple）：有效范围的尺度，例如 (0, np.inf)
    2. 函数功能：
       - 将预测框坐标转换为原始图像上的坐标
       - 剔除超出原始图像范围的部分
       - 将无效的边界框坐标设置为0
       - 剔除不在有效范围内的边界框
       - 剔除得分低于阈值的边界框
       - 返回过滤后的边界框信息
    3. 返回值：
       - 过滤后的边界框信息（形状：(M, 6)，数据类型：float32）：每个边界框包括 (xmin, ymin, xmax, ymax, score, class)
    4. 总结：
       该函数用于过滤和转换模型预测的边界框信息，确保边界框在有效范围内，并去除置信度较低的边界框。主要应用在目标检测模型的后处理阶段。
    """
    def __convert_pred(
        self, pred_bbox, test_input_size, org_img_shape, valid_scale
    ):
        """
        Filter out the prediction box to remove the unreasonable scale of the box
        """
        # (1)将预测框转换为原始图像上的坐标
        pred_coor = xywh2xyxy(pred_bbox[:, :4])  # [x, y, w, h] to [x1, y1, x2, y2]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        org_h, org_w = org_img_shape
        
        # （2）计算resize和裁剪的参数
        resize_ratio = min(
            1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h
        )
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # （3）裁剪掉超出原始图像范围的部分
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )
        # （4）将无效的边界框坐标设置为0
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]),
            (pred_coor[:, 1] > pred_coor[:, 3]),
        )
        pred_coor[invalid_mask] = 0

        # （5）剔除不在有效范围内的边界框
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
        )
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
        )

        # （6）剔除得分低于置信度阈值的边界框
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        # 组合有效的边界框信息
        mask = np.logical_and(scale_mask, score_mask)
        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
        )

        return bboxes

    """
    1. 入参：
       - `iou_thresh`（数据类型：float，默认值：0.5）：IoU（交并比）阈值，用于判定真正例的阈值
       - `use_07_metric`（数据类型：bool，默认值：False）：是否使用 VOC 2007 的评估指标
    2. 函数功能：
       - 根据预测结果和标注信息计算每个类别的平均精度（Average Precision，AP）
    3. 返回值：
       - 包含每个类别AP值的字典（数据类型：dict{cls: ap}）
    4. 总结：
       该函数通过调用voc_eval.voc_eval函数，计算每个类别的平均精度（AP）。函数依赖于预测结果文件、标注文件以及测试集的信息，通过设置IoU阈值和评估指标来进行评估。
    """
    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        Calculate ap values for each category
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        # 预测结果文件的格式化字符串
        filename = os.path.join(
            self.pred_result_path, "comp4_det_test_{:s}.txt"
        )
        # 缓存目录
        cachedir = os.path.join(self.pred_result_path, "cache")
        # annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        # 标注文件的格式化字符串
        annopath = os.path.join(
            self.val_data_path, "Annotations\\" + "{:s}.xml"
        )
        # 测试集文件路径
        imagesetfile = os.path.join(
            self.val_data_path, "ImageSets", "Main", "test.txt"
        )
        
        # 存储每个类别的Recall、Precision和AP
        APs = {}
        Recalls = {}
        Precisions = {}
        
        # 遍历每个类别
        for i, cls in enumerate(self.classes):
            # 调用voc_eval.voc_eval计算Recall、Precision和AP
            R, P, AP = voc_eval.voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                iou_thresh,
                use_07_metric,
            )
            Recalls[cls] = R
            Precisions[cls] = P
            APs[cls] = AP
            
        # 清理缓存目录
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
