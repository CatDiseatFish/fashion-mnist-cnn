# -*- coding: utf-8 -*-
"""
项目名称：基于深度学习的 Fashion-MNIST 服装图像分类
作者：23111301046 陈子航
说明：
1. 自动下载 Fashion-MNIST 数据集
2. 训练卷积神经网络 CNN
3. 输出训练/验证准确率和损失曲线
4. 输出测试集混淆矩阵
5. 输出分类报告
6. 可视化部分预测结果
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


# =========================
# 1. 基础参数设置
# =========================
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_DIR = "./data"
RESULT_DIR = "./results"
MODEL_PATH = os.path.join(RESULT_DIR, "fashion_mnist_cnn.pth")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# =========================
# 2. 固定随机种子，保证实验可复现
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. 定义 CNN 模型
# =========================
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B, 1, 28, 28] -> [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # -> [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # -> [B, 64, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 4. 训练一个 epoch
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


# =========================
# 5. 验证/测试一个 epoch
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# =========================
# 6. 画曲线图
# =========================
def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    epochs_range = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, val_losses, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 准确率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_accs, marker='o', label='Train Accuracy')
    plt.plot(epochs_range, val_accs, marker='s', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()


# =========================
# 7. 画混淆矩阵
# =========================
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


# =========================
# 8. 可视化预测结果
# =========================
def visualize_predictions(model, dataset, device, class_names, save_dir, num_images=16):
    model.eval()

    indices = np.random.choice(len(dataset), num_images, replace=False)

    plt.figure(figsize=(12, 12))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

            img = image.squeeze().cpu().numpy()

            plt.subplot(4, 4, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"T:{class_names[label]}\nP:{class_names[pred]}", fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_visualization.png"), dpi=300)
    plt.close()


# =========================
# 9. 主函数
# =========================
def main():
    set_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练集和测试集
    full_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # 从训练集中再划分出验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 模型、损失函数、优化器
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0

    print("\n开始训练...\n")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print("\n训练完成！")
    print(f"最佳验证集准确率: {best_val_acc:.4f}")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\n测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Accuracy: {test_acc:.4f}")

    # 输出分类报告
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n分类报告：")
    print(report)

    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # 生成图像
    plot_curves(train_losses, val_losses, train_accs, val_accs, RESULT_DIR)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, RESULT_DIR)
    visualize_predictions(model, test_dataset, device, CLASS_NAMES, RESULT_DIR, num_images=16)

    print("\n结果文件已保存到 results 文件夹，包括：")
    print("1. loss_curve.png")
    print("2. accuracy_curve.png")
    print("3. confusion_matrix.png")
    print("4. prediction_visualization.png")
    print("5. classification_report.txt")
    print("6. fashion_mnist_cnn.pth")


if __name__ == "__main__":
    main()
