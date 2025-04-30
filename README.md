# 训练模型
```bash
# 训练
python train.py -f config/ppo.yaml

# 查看训练进度
tensorboard --logdir=./logs/tensorboard --bind_all
```

# 测试与绘图
```bash
python main.py
```
