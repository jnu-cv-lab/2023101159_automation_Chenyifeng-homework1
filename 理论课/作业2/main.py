import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ====================== 1. 手动实现均值滤波（自行实现模块） ======================
def manual_mean_filter(img, kernel_size=3):
    """手动实现均值滤波（不调用cv2.blur）"""
    h, w = img.shape[:2]
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='edge')
    result = np.zeros_like(img, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            result[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
    return np.clip(result, 0, 255).astype(np.uint8)

# ====================== 2. 加载测试图像（3幅：低对比度、噪声、正常） ======================
def load_test_images():
    # 图像1：低对比度
    img1 = cv2.imread("low_contrast.jpg", 0)
    if img1 is None:
        img1 = np.random.randint(50, 150, (256, 256), dtype=np.uint8)

    # 图像2：含噪声
    img2 = cv2.imread("noisy.jpg", 0)
    if img2 is None:
        img2 = cv2.imread("test.jpg", 0)
        if img2 is None:
            img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        noise = np.random.normal(0, 20, img2.shape).astype(np.int16)
        img2 = np.clip(img2 + noise, 0, 255).astype(np.uint8)

    # 图像3：正常图像
    img3 = cv2.imread("test.jpg", 0)
    if img3 is None:
        img3 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    return [img1, img2, img3], ["低对比度", "含噪声", "正常"]

# ====================== 3. 图像增强方法 ======================
def enhance_methods(img):
    results = {}
    results["global_eq"] = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    results["clahe"] = clahe.apply(img)
    results["manual_mean"] = manual_mean_filter(img, 3)
    results["gaussian"] = cv2.GaussianBlur(img, (3,3), 0)
    results["median"] = cv2.medianBlur(img, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    results["sharpen"] = cv2.filter2D(img, -1, kernel)
    filter_then_eq = cv2.equalizeHist(manual_mean_filter(img, 3))
    results["filter->eq"] = filter_then_eq
    eq_then_filter = manual_mean_filter(cv2.equalizeHist(img), 3)
    results["eq->filter"] = eq_then_filter
    return results

# ====================== 4. 定量评价 ======================
def evaluate(img_original, img_processed):
    psnr_val = psnr(img_original, img_processed)
    ssim_val = ssim(img_original, img_processed, data_range=255)
    return psnr_val, ssim_val

# ====================== 5. 画图（已修复：4×3 子图） ======================
def plot_results(img, results, img_name):
    plt.figure(figsize=(18, 16))
    plt.subplot(4, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"原始图像：{img_name}")
    plt.axis("off")

    plt.subplot(4, 3, 2)
    plt.hist(img.flatten(), bins=256, color="gray")
    plt.title("原始直方图")

    methods = list(results.keys())
    for i, method in enumerate(methods):
        plt.subplot(4, 3, i + 3)
        plt.imshow(results[method], cmap="gray")
        plt.title(method)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"result_{img_name}.png")
    plt.close()

# ====================== 6. 主函数 ======================
if __name__ == "__main__":
    imgs, names = load_test_images()
    for img, name in zip(imgs, names):
        print(f"\n处理图像：{name}")
        results = enhance_methods(img)
        plot_results(img, results, name)
        print("定量评价指标（PSNR/SSIM）：")
        for method, res in results.items():
            p, s = evaluate(img, res)
            print(f"{method:12s} | PSNR: {p:.2f} dB | SSIM: {s:.4f}")
    print("\n✅ 全部完成！结果已保存为图片！")