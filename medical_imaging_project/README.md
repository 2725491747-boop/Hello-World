# 医学成像识别 PyTorch 项目

这个项目提供了一个最小可运行的医学成像识别骨架，使用 PyTorch 搭建简单的图像分类模型。

## 目录说明
- main.py: 训练与推理入口
- requirements.txt: 依赖列表
- sample_images/: 放置医学图像样本

## 安装依赖
在项目目录下执行：

```bash
pip install -r requirements.txt
```

## 运行
```bash
python main.py
```

## 说明
- 图片文件名中包含 normal 会被视为正常样本，包含 abnormal 会被视为异常样本。
- 你可以把自己的医学影像数据放进 sample_images 目录后继续训练。
