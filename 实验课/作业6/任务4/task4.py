import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务4：目标定位 =====================
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取灰度图像
img_box = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('box_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if img_box is None or img_scene is None:
    raise FileNotFoundError("错误：请确保图片在代码同一目录下！")

# 2. ORB 特征检测 + 描述子计算
orb = cv2.ORB_create(nfeatures=1000)
kp_box, des_box = orb.detectAndCompute(img_box, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 3. 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_box, des_scene)
matches = sorted(matches, key=lambda x: x.distance)

# 4. 提取匹配点坐标
src_pts = np.float32([kp_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 5. RANSAC 计算单应矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 6. 获取模板图像的四个角点
h, w = img_box.shape
box_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 7. 投影到场景图像，得到目标位置
scene_corners = cv2.perspectiveTransform(box_corners, H)

# 8. 绘制红色定位框
img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
img_result = cv2.polylines(img_scene_color, [np.int32(scene_corners)],
                          isClosed=True, color=(0, 0, 255), thickness=3)

# 9. 保存结果图片
cv2.imwrite('target_localization_result.png', img_result)

# 10. 显示结果
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.title('任务4：目标定位结果')
plt.axis('off')
plt.show()

