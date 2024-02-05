| 子数据集 | FD001 | FD002 | FD003 | FD004 |
| --- | --- | --- | --- | --- |
| 工作条件数量 |  1  |  6  |  1  |  6  |
| 故障模式数量 |  1  |  1  |  2  |  2  |
| 训练集样本数 | 100 | 260 | 100 | 249 |
| 测试集样本数 | 100 | 259 | 100 | 248 |


实验场景

数据集由多个多元时间序列组成。每个数据集又分为训练子集和测试子集。每个时间序列都来自不同的发动机，即数据可视为来自同一类型的发动机。每台发动机在开始时都有不同程度的初始磨损和制造变化，而这些对于用户来说都是未知的。这种磨损和变化被认为是正常的，即不被视为故障状况。**有三种操作设置会对发动机性能产生重大影响。这些设置也包含在数据中。数据受到传感器噪声的污染**。

发动机在每个时间序列开始时运行正常，在序列中的某个时刻出现故障。在训练集中，故障会逐渐加重，直至系统故障。在测试集中，时间序列在系统故障前一段时间结束。比赛的目的是预测测试集中故障发生前的剩余运行周期数，即发动机在最后一个运行周期后还能继续运行的运行周期数。同时提供测试数据的真实剩余使用寿命 (RUL) 值向量。

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by **spaces**. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

1)	unit number             # 发动机编号
2)	time, in cycles         # 时间，以工作循环数计
3)	operational setting 1   # 操作设置 1
4)	operational setting 2   # 操作设置 2
5)	operational setting 3   # 操作设置 3
6)	sensor measurement  1   # 传感器 1
7)	sensor measurement  2   # 传感器 2
    ...
26)	sensor measurement  26  # 传感器 26
