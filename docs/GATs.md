# GATs

Graph Attention networks

## Abstract

### GCN's limitation

GCN适用于处理transductive任务（即：训练和测试阶段基于同样的图结构，静态图），但它存在2个主要缺点：

* 无法完成 **inductive任务**（即：训练阶段与测试阶段需要处理的graph不同，动态图）
* 难以分配不同的weight给不同的neighbor

### GAT’s solution

* Global Graph Attention：每个顶点与图中所有顶点都进行attention运算
  * 优：完全不依赖于图结构
  * 缺：完全丢弃了图结构带来的信息 & 计算成本高
* Mask Graph Attention：注意力机制仅在邻居点上进行 ---> usually used



## Method

### Attention Coefficient

* 计算注意力系数

* 对每个顶点 i ，计算其邻居 j 与自己之间的相似系数
  * W：权重矩阵，对点信息增维（特征增强）
  * || ：将增强后的特征进行拼接
  * a ：将拼接后的特征映射到实数上

$$
e_{ij} = a([Wh_i||Wh_j]), j \in \mathcal{N}_i
$$

* 归一化：**Softmax**

```math
a_{ij} = \frac
{exp(LeakyReLU(e_{ij}))}
{\sum_{k \in \mathcal{N}_i}exp(LeakyReLU(e_{ik})) }
```

### Aggregate

* 加权求和
* Simple：

$$
h_i' = \sigma(\sum_{j\in \mathcal{N_i}} a_{ij}Wh_i)
$$

* with Multi-Heads:

$$
h_i' = \parallel_{k=1}^K  \sigma(\sum_{j\in \mathcal{N_i}} a_{ij}^kW^kh_i)
$$



## Analysis

* **GCN和GAT实际上都将邻居顶点的特征聚集于中心点**，但相比于GCN，GAT 往往更佳：考虑相邻点之间的**相关性**
* 为何GAT可用于Inductive任务？

> GAT 中的可学习参数为：W 和 a()，即如何对点信息进行增强，以及如何将增强信息映射到实数上，故：训练过程与图结构无直接关联，可适应不同图结构



## Appendix

### LeakyReLU

```math
LeakyReLU(x) = 
\left\{\begin{array}{ll}
x & \text { if } x \geq 0 \\
\alpha x & \text { if } x<0
\end{array}\right.
```

* LeakyReLU在输入值为正时，输出与ReLU相同，但**在输入值为负时，输出一个小的线性值**

* 适用场景：针对“死亡”神经元，在负输入情况下，LeakyReLU仍然有一个小的梯度（$\alpha $），使其权重在训练过程中仍然可以更新
