### Into

该项目利用图神经网络等技术，预测未知抗原-抗体对之间的中和效果

模型主要包括3部分：

* 构建抗原抗体的图结构并提取embedding
* 构建二分图
* 依据二分图预测中和效果

### Mywork
因为实验室之前利用后半部分的结构在药物-病毒治疗效果任务中取得了很好的效果，所以我在对模型结构复现后，主要工作在于从氨基酸链中提取embedding，并依据embedding构建二分图：

* 小学期前期工作：相关基础理论学习，理解之前类似工作（MCGCL-DTA）的代码，并整理并规范化了原来的代码
  * 大白话：模型的后2步是基于实验室学姐的方法去做的，但她的代码写得不好，我整理了一下，然后在生成二分图这一步，做了适配性的修改，但现在的生成二分图的代码好像还是原来的

* 小学期后期工作：基于DeepAAI模型，将其应用于本项目
  * 大白话：我整理了DeepAAI模型的思路，尤其是维度变化方面，然后修改模型，实现对抗原抗体氨基酸链的特征提取，除此之外，我研究了DeepAAI模型对氨基酸链输入的处理方式，在假期的时候尝试做了一些修改

* 在准备进行训练的时候，发现由于原方法在构建二分图这一步实际上是从文件中读取提取好的特征，如果不对构建二分图这一步做修改的话，那模型就不是端到端训练。当时想到了2种解决方法：
  1. 修改构造二分图代码，将模型改为端到端
  2. 先用AE方式训练前部分，提取特征，再做后半部分
* 然后和老师稍微讨论了一下，我假期尝试了第二种方法，结果：大概是当前结果不如纯DeepAAI结构的预测方法，而且因为当前模型结构并非端到端的结构，我无法确定我提embedding的质量
  * 除此之外，因为我觉得纯AE还原整个氨基酸链，实际上并不是面向抗原抗体功能特性的。了解到氨基酸链中实际上具有功能特性的只有部分，所以我尝试了在AE方法中，decode出氨基酸的功能部分。因为找到的工具比较麻烦，所以只进行了小样本的实验，效果不确定，因为非端到端的话，我无法确定是前部分出问题还是后部分出问题

* 未来：我感觉还是改成端到端的好


### 如何修改DeepAAI方法

原deepAAI结构：结合抗原和抗体各自的global_ft和local_ft产生抗体对的global和local，再结合抗体对的global和local，实现预测

因为，模型任务由预测变为encode，需要分别生成抗原、抗体的embedding，所以进行修改

修改后的结构：直接结合抗原和抗体的global和local，分别生成embedding

### 输入模式

* kmer：将氨基酸链分为长度为k的子序列，来捕获局部特征，通过比较kmer的频率信息来预测功能
* pssm：对氨基酸链进行多序列比对，计算每一个位置上某个氨基酸出现的概率，实现复杂的模式识别，捕获序列中的特定模式，尤其是特别位点的信息