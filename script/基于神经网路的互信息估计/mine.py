# -*- coding: utf-8 -*-
"""
Created on 2024/04/08 15:17:36

@File -> mine.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Mutual Information Neural Estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 定义两个全连接层，第一个有10个节点，第二个有5个节点
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)  # 输出层有2个节点，对应二分类问题

    def forward(self, x):
        # 定义正向传播，添加ReLU激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
if __name__ == "__main__":
    # 实例化模型并将其移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 随机生成一些数据进行训练
    for t in range(100):
        # 生成随机输入和目标
        input = torch.randn(10, 20).to(device)  # 10个样本，每个样本20个特征
        target = torch.randint(0, 2, (10,)).to(device)  # 每个样本的目标是0或1
    
        # 前向传播
        output = model(input)
        loss = criterion(output, target)
    
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if t % 10 == 0:
            # 每10个batch打印一次loss
            print(f"t={t}, loss={loss.item()}")
    
    # 完成训练后，我们可以用模型做一些预测
    inputs = torch.randn(2, 20).to(device)
    prediction = model(inputs)
    print("Predictions:", prediction)