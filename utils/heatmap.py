import cv2
import math
import random
import numpy as np
import os



"""
1. 入参：
   - `color`: 彩色数组，形状为 (3,)
   - `clmsk`: 彩色遮罩张量，形状为 (h, w, 3)，数据类型为 float32
   - `mskd`: 灰度遮罩张量，形状与 `clmsk` 相同，数据类型为 float32
   - `beta`: 注意力热图张量，形状为 (1, i, h1 * w1)，数据类型为 float32
   - `h1`, `w1`: 注意力热图的高度和宽度
   - `img_show`: 叠加遮罩后的图像张量，形状与 `clmsk` 相同，数据类型为 float32

2. 函数功能：
   - 调整彩色遮罩的颜色
   - 更新叠加彩色遮罩后的图像
   - 处理注意力热图，生成热图并与原始图像叠加
   - 保存生成的热图和叠加遮罩后的图像

3. 返回值：
   - 无返回值，但将生成的热图和叠加遮罩后的图像保存为文件

4. 总结：
   该函数用于调整颜色、更新图像、处理注意力热图，并保存结果。主要应用在模型可视化的场景，帮助理解模型对图像的关注区域。
"""
def imshowAtt(beta, img=None):
    # 创建两个窗口用于显示图像
    cv2.namedWindow("img")
    cv2.namedWindow("img1")
    assert img is not None

    # 复制原始图像以防止修改
    h, w, c = img.shape
    img1 = img.copy()
    img = np.float32(img) / 255  # 对复制的图像进行归一化处理
    
    # 修改注意力图的形状
    (height, width) = beta.shape[1:]  # beta代表注意力热图
    # 对高宽进行开平方操作
    h1 = int(math.sqrt(height))
    w1 = int(math.sqrt(width))

    # 对每个注意力图进行处理
    for i in range(height):
        # 复制原始图像用于显示
        img_show = img1.copy()
        h2 = int(i / w1)
        w2 = int(i % h1)

        # 创建一个二维的mask，用于高亮注意力的位置
        mask = np.zeros((h1, w1), dtype=np.float32)
        mask[h2, w2] = 1  # 在数组中对应mask位置上将值设置为1
        # 将mask调整为与原始图像相同的大小
        mask = cv2.resize(mask, (w, h))
        # 将mask扩展为3个通道，沿着第三个维度进行重复，用于与图像相乘
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # 通过将原始图像与mask相乘，得到高亮了注意力位置的图像
        mskd = img_show * mask
        
        # 生成一个随机颜色，用于增强高亮效果
        color = (random.random(), random.random(), random.random())
        # 创建一个形状与mask相同的全1数组，并乘以颜色值和256， 用于增加颜色色度
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        # 将生成的颜色 mask 叠加到原始图像上，并进行权重调整
        img_show = img_show + 0.8 * clmsk - 0.8 * mskd
        
        # 从 beta 中提取第 0 个通道的第 i 个位置的值，即注意力权重
        # .view()：pytorch调整张量大小的方法；.resize()：opencv库调整数组大小的方法
        cam = beta[0, i, :]
        # 将 cam 转换为二维数组形状 (h1, w1)，并转为 NumPy 数组
        cam = cam.view(h1, w1).data.cpu().numpy()
        # 调整 cam 的大小为原始图像的大小
        cam = cv2.resize(cam, (w, h))
        # 将cam整体减去其中的最小值， 是整体数值平移非负
        cam = cam - np.min(cam)
        # 归一化
        cam = cam / np.max(cam)
        # cam = 1 / (1 + np.exp(-cam))

        # 使用OpenCV将灰度图cam应用伪彩色映射（热力图）
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # heatmap转化为浮点型并且进行归一化到[0, 1]区间
        heatmap = np.float32(heatmap) / 255
        # 将彩色热力图与原始图像叠加
        cam = heatmap + np.float32(img)
        # 像素值整体平移，像素值非负操作
        cam = cam - np.min(cam)
        # 叠加后的图像像素值归一化
        cam = cam / np.max(cam)
        # 将图像浮点型像素值转化为整型，并保存为图片文件
        cam = np.uint8(255 * (cam))
        cv2.imwrite("att.jpg", cam)
        # 保存原始图像加上注意力热图的结果
        cv2.imwrite("img.jpg", np.uint8(img_show))
        # 在窗口中显示注意力热图
        cv2.imshow("img", cam)
        # 显示原始图像加上热图的结果
        cv2.imshow("img1", np.uint8(img_show))
        # 等待按键事件， 按下“q”键关闭窗口并退出程序
        k = cv2.waitKey(0)
        if k & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit(0)
