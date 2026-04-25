import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务1：ORB 特征点检测 =====================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图像
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 判断图像是否读取成功
if img_box is None or img_scene is None:
    raise FileNotFoundError("错误：请确保图片在代码同一目录下！")

# 2. 创建 ORB 特征检测器
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点 + 计算描述子
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 4. 绘制关键点（带方向和大小，更专业）
img_box_kp = cv2.drawKeypoints(img_box, kp_box, None, (0, 255, 0),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_scene_kp = cv2.drawKeypoints(img_scene, kp_scene, None, (0, 255, 0),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 5. 保存结果图片
cv2.imwrite('box_keypoints.png', img_box_kp)
cv2.imwrite('box_in_scene_keypoints.png', img_scene_kp)

# 6. 输出实验结果
print("===== 任务1 ORB 特征点检测结果 =====")
print(f"box 关键点数量：{len(kp_box)}")
print(f"scene 关键点数量：{len(kp_scene)}")
print(f"描述子维度：{des_box.shape[1]}")

# 7. 显示图像
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img_box_kp), plt.title('box 特征点')
plt.subplot(122), plt.imshow(img_scene_kp), plt.title('scene 特征点')
plt.show()