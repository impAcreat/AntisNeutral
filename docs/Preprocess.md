# Preprocess

## Abstract

DeepAAI提供的解决方案中，预处理包括2步：

1. 处理原始xlsx文件，处理数据，提取所需内容并存储在文件中（processing目录的作用）
2. 读取1中生成的文件，构造abs_dataset类，用于模型

可见，关键在于processing



## Processing

* 目录结构：

```
processing/
│
├── hiv_cls/
│   ├── corpus/						# processing 结果
│   │   ├── cls/...					
│   │   ├── processed_mat/...
│   ├── processing.py				# 主函数
│   ├── dataset_tools.py			# 数据集的处理工具类
│   ├── feature_trans_content.py	# 氨基酸字典
│   ├── k_mer_utils.py				# kmer 工具类
│   └── dataset_hiv_cls.xlsx		# 原数据
│
├── cov_cls/...		# 类似
├── hiv_reg/...		# 类似
```

****

### processing

* 处理原文件，生成所需内容

* input：xlsx file

| column       | value         | meaning      |
| ------------ | ------------- | ------------ |
| antibody_seq | string        | 抗体氨基酸链 |
| virus_seq    | string        | 病毒氨基酸链 |
| label        | 0 / 1         | 中和效果     |
| split        | seen / unseen | 结点是否已知 |

* output：

| file                     | function                                               |
| ------------------------ | ------------------------------------------------------ |
| train_index              | seen 结点的索引 （complete）                           |
| test_unseen_index        | unseen 结点的索引（complete）                          |
| all_label_mat            | label （complete）                                     |
| antibody_index_in_pair   | complete 集合中抗体对应的 unique集合索引（complete）   |
| virus_index_in_pair      | ......                                                 |
| known_antibody_idx       | seen抗体的 unique 集合索引                             |
| unknown_antibody_idx     | ......                                                 |
| known_virus_idx          | ......                                                 |
| raw_all_antibody_set_len | complete集合中氨基酸长度（complete）                   |
| raw_all_virus_set_len    | ......                                                 |
| **protein_ft_dict**      | 特征矩阵字典（包括抗体和病毒各类表征方式对应的ft_mat） |

* `protein_seq_list_to_ft_mat` 函数：基于**氨基酸-表征字典**，通过一一对应将氨基酸序列转为特征矩阵
  * 输入：
    * protein_seq_list
    * max_seq_len
    * ft_type：目标特征矩阵的类型（one-hot / phych / amino_num）
  * 输出：ft_mat
* **ft_mat_dict**：
  * 当前的特征矩阵包括：
    * one-hot：一个氨基酸对应一个one-hot（故一个链对应一个矩阵）
    * amino_num：一个氨基酸对应一个索引（故一个链对应一个vector）
    * kmer：一个链对应一个kmer特征向量




****

### dataset_tools

* 处理数据集的工具函数
* `get_padding_ft_dict` 函数：基于氨基酸字典文件（feature_trans_content.py），生成**氨基酸-表征字典**
  * 输出：
    * one-hot ft
    * pssm ft
    * physicochemical ft
    * pad_amino_map_idx：氨基酸-索引 map



----

### feature_trans_content

* 氨基酸字典文件，将氨基酸与索引或表征对应
	- amino_map_idx: map amino acid to index
  - amino_pssm_ft: amino acid pssm feature
  - amino_physicochemical_ft: amino acid physicochemical feature



****

### kmer

* 基于词频的特征矩阵

* `KmerTranslator`类，重点在于 fit_transform，算法如下

  >1. 将蛋白质的字符串转为 k-mer 表示形式（每个字符被转换为相应的索引，并以空格分隔）
  >2. 学习k-mer特征（基于词频），并将数据转换为矩阵
  >
  >3. 标准化处理
  >4. 转换：提供了2种转换（标准转换std 和 概率转换prob）
  >5. 返回转换结果



