# Fashion-MNIST CNN Classification

## 项目简介
本项目基于 PyTorch 实现卷积神经网络（CNN），对 Fashion-MNIST 数据集进行服装图像分类。

## 数据集
- Fashion-MNIST
- 10类服装
- 训练集：60000
- 测试集：10000

## 模型结构
- 2层卷积
- ReLU激活
- MaxPooling
- 全连接层 + Dropout

## 实验结果
- 测试集准确率：91.41%

## 使用方法
```bash
pip install -r requirements.txt
python main.py
