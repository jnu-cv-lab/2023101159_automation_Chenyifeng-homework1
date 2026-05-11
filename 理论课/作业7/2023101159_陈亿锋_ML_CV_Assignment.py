import os
import warnings
from pathlib import Path

# 关闭警告，保证运行干净
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 固定随机种子，结果可复现
RANDOM_STATE = 42
OUTPUT_DIR = Path("assignment_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================
# 任务1：样本图像保存
# ==============================
def save_sample_images(images, labels, save_path):
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.suptitle("Digits Dataset Samples (Task 1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# ==============================
# 任务6：混淆矩阵
# ==============================
def save_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({model_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return cm

# ==============================
# 任务6：错误样本图
# ==============================
def save_misclassified_examples(images_test, y_true, y_pred, save_path, max_num=12):
    err_idx = np.where(y_true != y_pred)[0]
    select_idx = err_idx[:max_num]
    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(select_idx):
        plt.subplot(3, 4, i+1)
        plt.imshow(images_test[idx], cmap="gray")
        plt.title(f"True:{y_true[idx]} Pred:{y_pred[idx]}")
        plt.axis("off")
    plt.suptitle("Misclassified Examples (Task 6)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return err_idx

# ==============================
# 主程序：严格按作业任务执行
# ==============================
def main():
    print("=" * 60)
    print("      手写数字图像分类实验（传统机器学习）")
    print("=" * 60)

    # ========== 任务1：数据准备 ==========
    digits = load_digits()
    images = digits.images
    X = digits.data
    y = digits.target

    print("\n【任务1：数据准备】")
    print(f"图像总数：{len(images)} 张")
    print(f"图像尺寸：{images.shape[1]} × {images.shape[2]} 像素")
    print(f"类别标签：{sorted(list(set(y)))}")
    print(f"特征形状：{X.shape}（每张图展平为64维向量）")
    save_sample_images(images, y, OUTPUT_DIR / "task1_samples.png")
    print("样本图像已保存 → assignment_outputs/task1_samples.png")

    # ========== 任务2：数据划分（25%测试集） ==========
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X, y, images, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print("\n【任务2：数据划分】")
    print(f"训练集数量：{len(X_train)} 张（用于训练模型）")
    print(f"测试集数量：{len(X_test)} 张（用于评估泛化能力）")

    # ========== 任务3：特征表示 ==========
    print("\n【任务3：特征表示】")
    print("8×8图像 → 展平为一维64维特征向量")
    print("传统模型无法直接输入二维图像，必须转为向量格式")

    # ========== 任务4：模型训练（6个模型全覆盖） ==========
    models = {
        "KNN":                  make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Naive Bayes":          GaussianNB(),
        "Logistic Regression":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        "SVM":                  make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, random_state=RANDOM_STATE)),
        "Decision Tree":        DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    }

    print("\n【任务4 & 任务5：模型训练与准确率对比】")
    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))
        predictions[name] = y_pred
        print(f"{name:<22} 测试准确率：{acc:.4f}")

    # 按准确率排序
    results.sort(key=lambda x: x[1], reverse=True)
    best_model = results[0][0]
    best_acc = results[0][1]

    # 保存CSV表格（任务5）
    with open(OUTPUT_DIR / "task5_accuracy_table.csv", "w", encoding="utf-8") as f:
        f.write("模型,测试准确率\n")
        for n, a in results:
            f.write(f"{n},{a:.4f}\n")

    print(f"\n✅ 最优模型：{best_model}（准确率：{best_acc:.4f}）")
    print("✅ 准确率表格已保存 → task5_accuracy_table.csv")

    # ========== 任务6：错误样本分析（最优模型） ==========
    print("\n【任务6：错误样本分析】")
    best_y_pred = predictions[best_model]
    cm = save_confusion_matrix(y_test, best_y_pred, best_model, OUTPUT_DIR / "task6_confusion_matrix.png")
    err_indices = save_misclassified_examples(images_test, y_test, best_y_pred, OUTPUT_DIR / "task6_misclassified.png")

    print(f"混淆矩阵已保存 → task6_confusion_matrix.png")
    print(f"错误样本图已保存 → task6_misclassified.png")
    print(f"测试集错误总数：{len(err_indices)}")

    # 输出最容易混淆的数字
    print("\n最容易混淆的数字对（真实→预测）：")
    confusion_pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm[t, p] > 0:
                confusion_pairs.append((t, p, cm[t, p]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for t, p, cnt in confusion_pairs[:5]:
        print(f"  {t} → {p} ：{cnt} 次")

    print("\n" + "="*60)
    print("全部任务完成！所有输出在 assignment_outputs 文件夹")
    print("="*60)

if __name__ == "__main__":
    main()