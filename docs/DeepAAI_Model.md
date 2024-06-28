# DeepAAI Model

## Dimension

> Code: deep_aai_kmer_pssm_embedding_cls

### Layers(global ft)

* \__init__()

| layer                | in_features   | out _features | func |
| -------------------- | ------------- | ------------- | ---- |
| antibody_kmer_linear | kmer          | h             |      |
| virus_kmer_linear    | kmer          | h             |      |
| antibody_pssm_linear | antibody_pssm | h             |      |
| virus_pssm_linear    | virus_pssm    | h             |      |
| share_linear         | h             | h             |      |
| share_gcn1           | h             | h             |      |
| share_gcn2           | h             | h             |      |
| antibody_adj_trans   | h             | h             |      |
| virus_adj_trans      | h             | h             |      |
| global_linear        | 2*h           | h             |      |
| pred_linear          | h             | 1             |      |

> specific:
>
> * kmer_dim:  1893
> * h_dim:  512
> * pssm_antibody_dim:  840
> * pssm_virus_dim:  420

### Dim transformation(global ft)

* forward()

##### Step 1: get antibody feature and virus feature

* antibody transformation is similar to virus transformation, so take antibody as an example
* n = 197 in this case

> #### Combine input & Generate adj
> * process and combine kmer and pssm input
> * generate adj: adj is adjacency matrix, represent similarity between nodes
> 
> | input M           | input Dim | Layer \|\| func                    | output Dim | output M              |
> | ----------------- | --------- | ---------------------------------- | ---------- | --------------------- |
> | ft_dict_kmer      | [n, kmer] | antibody_kmer_linear               | [n, h]     | antibody_node_kmer_ft |
> | ft_dict_pssm      | [n, pssm] | antibody_pssm_linear               | [n, h]     | antibody_node_pssm_ft |
> | kmer & pssm       | [n, h]    | cat()-> activate -> dropout        | [n, 2*h]   | antibody_node_ft      |
> |antibody_node_ft | [n, 2*h] | share_linear-> act -> dropout | [n, h] | antibody_node_ft(res*) |
> | antibody_node_ft  | [n, 2*h]  | antibody_adj_trans -> tanh         | [n, h]     | antibody_trans_ft     |
> | antibody_trans_ft | [n, h]    | norm()                             | [n, 1]     | w                     |
> | w                 | [n, 1]    | w * w.t()                          | [n, n]     | w_mat                 |
> | trans_ft & w_mat  | [n, n]    | mm(trans_ft, trans_ft.t()) / w_mat | [n, n]     | antibody_adj          |
>

> #### GCN in main model
>
> * use 2 gcn to generate nodes representation based on graph
>
> | input M       | input Dim       | Layer \|\| func     | output Dim | output M      |
> | ------------- | --------------- | ------------------- | ---------- | ------------- |
> | adj & node_ft | [n, n] & [n, h] | share_gcn1          | [n, h]     | node_ft(res*) |
> |               |                 | ->activate->dropout |            |               |
> | adj & node_ft | [n, n] & [n, h] | share_gcn2          | [n, h]     | node_ft(res*) |
> 

> #### GCN conv
>
> * code path: /models/layers/gcn_conv_input_mat.py & model_utils.py
>* Basic
>   * input: node_mat & adj_mat
>   * Parameter: self.weight, [n, h]
> 
> * Method: 
>   * use laplace to process adj
>   * add trainable parameter to make the conv more adaptive
>
> 
>1. get laplace_mat: from adj_mat
> 2. node_state = mm(laplace_mat, node_mat)
>3. node_state = mm(node_state, self.weight)

> ### Add
>
> $$
> antibody\_res\_mat =\sum node\_ft
> $$
>
> * the added node_ft is pointed by '(res*)' in the tables

##### Step 2:  Cross and Predict

> ### Global feature
>
> * n' represent batch_size
>
> | input M          | input Dim   | Layer \|\| func | output Dim | output M       |
> | ---------------- | ----------- | --------------- | ---------- | -------------- |
> | antibody_res_mat | [n, h]      | [index]         | [n', h]    | res_mat        |
> | virus_res_mat    | ...         | ...             | ...        | ...            |
> | 2_res_mat        | 2 * [n', h] | cat             | [n', 2*h]  | global_pair_ft |
> |                  |             | -> activate     |            |                |
> |                  |             | -> dropout      |            |                |
> | global_pair_ft   | [n', 2*h]   | global_linear   | [n', h]    | global_pair_ft |

****



### Layers(local ft)

| Layer | in_features          | out_features |
| ----- | -------------------- | ------------ |
| CNN1  | antibody_max_len * l | h (512)      |
| CNN2  | virus_max_len * l    | h (512)      |



### Dim transformation(local ft)

* forward()

> ### CNN
>
> * use cnn to antibody/virus feature from their amino feature
>
> | input M       | input Dim         | Layer \|\| func | output Dim        | output M      |
> | ------------- | ----------------- | --------------- | ----------------- | ------------- |
> | (a/v)amino_ft | [n', max_len, l]  | view            | [n', max_len * l] | (a/v)amino_ft |
> | a_amino_ft    | [n', max_len * l] | cnn1            | [n', h]           | a_ft          |
> | v_amino_ft    |                   | cnn2            |                   | v_ft          |
> | a/v_ft        | 2 * [n', h]       | cat             | [n', 2*h]         | local_pair_ft |
> |               |                   |                 |                   |               |
>

>
> ### Local feature
> | input M       | input Dim | Layer \|\| func | output Dim | output M      |
> | ------------- | --------- | --------------- | ---------- | ------------- |
> |               |           | activate        |            |               |
> | local_pair_ft | [n', 2*h] | linear          | [n', h]    | local_pair_ft |
> |               |           | activate        |            |               |
> | local_pair_ft | [n', h]   | linear          | [n', h]    | local_pair_ft |
