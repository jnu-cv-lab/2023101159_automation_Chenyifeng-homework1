# Python视觉开发环境搭建与图像基本读写

## 实验内容
1. 使用 OpenCV 读取测试图片并输出图像基本信息（宽度、高度、通道数、数据类型）。
2. 显示原始图片，并将彩色图转换为灰度图后显示。
3. 保存灰度图为新文件，并使用 NumPy 对图像进行裁剪、获取指定位置像素值。

## 运行环境
- 系统：WSL Ubuntu 22.04
- Python 版本：3.8+
- 依赖库：opencv-python、numpy、matplotlib

## 基本信息
- 图像文件：test.jpg
- 图像基本信息：宽度: 600 像素 
- 高度: 400 像素 通道数: 3 
- 像素数据类型: uint8 
## 实验结果
- 灰度图已保存为: gray_test.jpg 灰
- 度图 (100,100) 
- 位置像素值: 58 
- 左上角 100x100 
- 区域图已保存为: cropped_test.jpg
## 运行方法
```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install opencv-python numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
