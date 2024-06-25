# Contrastive Learning

## Self-Supervised Learning

![在这里插入图片描述](https://img-blog.csdnimg.cn/81e7a450251b4071ac08d1b2e7f6ffb9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5a2m5rij5rij5rij5rij5rij,size_20,color_FFFFFF,t_70,g_se,x_16)

* **自监督学习**（self-supervised learning）不需要人工标注的类别标签信息，直接利用数据本身作为监督信息，学习样本数据的特征表达，应用于下游的任务

* 分类：
  * **对比学习**：通过将数据分别与正例样本和负例样本在特征空间进行对比，来学习样本的特征表示。主要的难点在于如何构造正负样本
  * **生成学习**：以自编码器为代表，主要关注pixel label的loss。例如，在自编码器中对数据样本编码成特征再解码重构，认为重构的效果比较好则说明模型学到了比较好的特征表达

* 对比学习的优势：生成学习注重样本特征，可能过于注重细节，而对比学习注重特征空间上的区分性



## Method

> 1. 生成Anchor的正负样本对
> 2. Encoder处理 ---->  Projection处理
> 3. Loss计算
> 4. 模型训练

#### Data Augmentation

* 生成不同视图或增强视图，用于构造正负样本

****

#### Encoder

* 编码器 将正样本（增强实例）映射到表征空间，捕获特征与相似性

****

#### Projection

* 投影网络获取编码器网络的输出并将其投影到低维空间
* 投影网络降低了数据的复杂性和冗余，有助于更好地分离相似和不相似的实例

****

#### Constrast Learning

* 目的：通过训练，使相似实例在嵌入空间中距离更近，而不相似实例距离更远

****

#### Loss(InfoNCE)

* 核心思想：最大化正样本对与最小化负样本对之间的相似性
* 假设我们有一个查询（query）向量 q 和一个正样本向量 k+，以及 K 个负样本向量 {k−}，InfoNCE损失可以定义为：

$$
\mathcal{L}_{\text {InfoNCE }}=-\log \frac{\exp \left(q \cdot k^{+} / \tau\right)}{\exp \left(q \cdot k^{+} / \tau\right)+\sum_{i=1}^{K} \exp \left(q \cdot k_{i}^{-} / \tau\right)}
$$

****

#### Loss(logistic Loss)

* 又名：逻辑回归损失或交叉熵损失
* 常用于分类问题

$$
\text { LogLoss }=\sum_{(x, y) \in D}-y \log \left(y^{\prime}\right)-(1-y) \log \left(1-y^{\prime}\right)
$$



