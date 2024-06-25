# GCNs

依据图的结构更改节点表征

## Graph 

### Tasks

- 结点分类(Node classification)：给定一个结点，预测其类型。
- 链路预测(Link prediction)：预测两个结点之间是否存在连接。
- 社区检测(Community detection)：确定具有紧密连接关系的结点簇。
- 网络相似度(Network similarity)：衡量两个网络或子网络之间的相似性。

### Info in Graph

图中的信息主要包括2部分：

* 节点特征
* 图的结构：节点间关系

## Methods

* **GCNs**：可直接利用图的结构信息，属于卷积神经网络
* **半监督学习**：使用部分带标签数据和大量未带标签的数据进行模型训练
  * 减少标注成本
  * 提升模型性能：充分利用未标记数据，获取更多训练信息，提高performance

* 目的：与CNNs相似，从图结构中**提取信息**

****

### Principle

* 图的表示：节点数量-N，（输入）节点特征维度-D
  * 节点特征矩阵 **X**：N*D
  * 节点关系矩阵 **A**：N*N

* GCN神经网络层的传播方式：

$$
H^{(l+1)} = \sigma(
\widetilde{D}^{-\frac{1}{2}}
\widetilde{A}
\widetilde{D}^{-\frac{1}{2}}
H^{(l)}
W
)
$$

> $\widetilde{A} = A + I$：$I$  为单位矩阵
>
> $\widetilde{D}_{ii} = \sum_j \widetilde{A}_{ij}$ ：为A的度矩阵
>
> $H$ ：是每一层的特征矩阵，输入层即为X
>
> $W$ ：可训练权重
>
> $\sigma$  ：非线性激活函数

* 损失函数：依据任务的不同而不同

****

### Understanding

* 图结构最直接的处理模型：

$$
f(H^{(l)}, A)=
\sigma(A H^{(l)} W^{(l)})
$$

* 依据邻接矩阵A提供的图结构信息，处理节点特征矩阵H，再乘权重矩阵，最后激活
  * 该模型已经具有提取特征的能力
  * 模型局限性：
    * 直接使用 $A$（而非 $\widetilde{A}$ ），在与 W 相乘时只考虑了相邻节点的特征，忽略了自己本身的特征
    * A未经过归一化处理，与特征矩阵相乘会改变特征原本的分布，可能产生一些不可预测的问题

* 改进：

$$
\widetilde{D}^{-\frac{1}{2}}
\widetilde{A}
\widetilde{D}^{-\frac{1}{2}}
$$

* 改进原理：

1. Step 1：$\widetilde{D}^{-1}\widetilde{A}$

   * 从上图可以看出，用 $\widetilde{D}^{-1}$ 处理相当于进行**加权平均**：加权平均的想法在于那些具有较低度的结点会对其邻居有较大的影响，而高度的结点会产生更小的影响因为他们的影响会被分散给很多的邻居结点

   * 仅对行进行了标准化，并未对列进行处理

<img src="https://topb0ts.wpenginepowered.com/wp-content/uploads/2020/10/1_cfx8c0lEAm7RQEl-Jn19uA.png" alt="img" style="zoom:80%;" />

<img src="https://topb0ts.wpenginepowered.com/wp-content/uploads/2020/10/1_o7jb9GT_ePKE6c9hHy9XEw.png" alt="img" style="zoom:67%;" />

2. Step 2：$\widetilde{D}^{-1}\widetilde{A}\widetilde{D}^{-1}$，添加对列的标准化

   * 两次标准化将值缩小过多

   ![graph convolutional network](https://topb0ts.wpenginepowered.com/wp-content/uploads/2020/10/1_FLQi-BEivJgztalgq7xIOwedited.png)

3. Step 3：$\widetilde{D}^{-\frac{1}{2}} \widetilde{A} \widetilde{D}^{-\frac{1}{2}} $

<img src="https://topb0ts.wpenginepowered.com/wp-content/uploads/2020/10/1_cfx8c0lEAm7RQEl-Jn19uA.png" alt="img" style="zoom:80%;" />



## Performance

* 优秀
* 即使不进行训练，使用随机初始化的参数，都能获得优秀的结果



## More

* 原理详解：https://blog.csdn.net/qq_26593695/article/details/109393775
