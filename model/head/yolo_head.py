import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors  # YOLO模型的锚点
        self.__nA = len(anchors)  # 锚点的数量
        self.__nC = nC  # 类别数目
        self.__stride = stride  # 步长

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        # 将输入张量进行形状变换和维度交换，以便后续处理
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
        
        # 对变换后的张量进行解码
        p_de = self.__decode(p.clone())

        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
        
        # 张良（p）张量的维度说明：
        # p.shape: (batch_size, num_anchors, num_outputs, grid_size, grid_size)
        # shape: (batch_size, num_anchors, num_outputs, grid_size, grid_size, 2)
        # 第一个通道对应 x 坐标的偏移，第二个通道对应 y 坐标的偏移
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        
        # shape: (batch_size, num_anchors, num_outputs, grid_size, grid_size, 2)
        # 第一个通道对应宽度的信息，第二个通道对应高度的信息
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        
        # shape: (batch_size, num_anchors, num_outputs, grid_size, grid_size, 1)
        # 单通道，对应模型对边界框的置信度
        conv_raw_conf = p[:, :, :, :, 4:5]
        
        # shape: (batch_size, num_anchors, num_outputs, grid_size, grid_size, num_classes)
        # 包含每个类别的概率信息，num_classes 是类别的数量
        conv_raw_prob = p[:, :, :, :, 5:]

        """
        创建一个行向量，表示y坐标， 然后通过unsqueeze在第二维度上添加一个维度，使其变为列向量；
        最后通过repeat在第一维度上进行复制，得到一个网格矩阵
        """
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        
        """
        创建一个类向量，表示x坐标，然后通过unsqueeze在第一维度上添加一个维度，使其变为行向量；
        最后通过repeat在第二维度上进行复制，得到一个网格矩阵
        """
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        
        """
        将x和y拼接在一起，得到一个形状为（output_size, output_size, 2）的网格；
        这里的2表示每个点的坐标是一个二维向量，即（x, y）
        """
        grid_xy = torch.stack([x, y], dim=-1)
        
        """
        在网格矩阵的前面添加两个维度，得到一个形状为（batch_size,output_size, 3， 2）的张量；
        这里的3表示每个格子有三个坐标， 对应每个锚点的预测；
        然后通过repeat在第一维度上进行复制，得到与预测张量相同的batch_size;
        最后将张量转换为float类型，并移到指定的设备（device）
        """
        grid_xy = (
            grid_xy.unsqueeze(0)
            .unsqueeze(3)
            .repeat(batch_size, 1, 1, 3, 1)
            .float()
            .to(device)
        )

        """
        解码预测的边界框中心坐标：
        对预测的x和y坐标应用sigmoid函数，然后加上之前构建的网格信息，再乘以步长；
        得到最终的边界框宽度和高度的预测。
        """
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        
        """
        解码预测的边界框宽度和高度：
        对预测框的宽的和高度应用指数函数，然后乘以锚点信息，再乘以步长；
        得到最终的边界框宽度和高度的预测
        """
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        
        """
        将坐标中心和宽高信息拼接在一起，得到包含中心坐标，宽高的坐标张量
        """
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        
        """
        解码边界框置信度：
        对预测的置信度应用sigmoid函数
        """
        pred_conf = torch.sigmoid(conv_raw_conf)
        
        """
        解码预测的边界框类别概率：
        对预测的概率类别应用sigmoid函数
        """
        pred_prob = torch.sigmoid(conv_raw_prob)
        
        """
        将解码后的结果拼接为最终的边界框预测张量；
        包含了最终的边界框坐标、置信度分数和类别概率的预测张量
        """
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return (
            pred_bbox.view(-1, 5 + self.__nC)  # 对非训练模式的输出张量进行调整形状
            if not self.training
            else pred_bbox   # 训练模式的输出为原始输出
        )
