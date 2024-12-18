## Abstract
* This repository is focusing on developing a model that can predict neutralization effect of antibody.
* based on GCNs, GATs, MLP

****

## Docs
**Include notes and code analysis**
* basic knowledge: GATs, GCNs...

* DeepAAI:
  * DeepAAI.md: note of original paper
  * DeepAAI_Model.md: understanding of original DeepAAI model & dimension change
  * DeepAAI_Modification.md: my modification to model

* Preprocess: conclusion of the way to preprocess antis

****

## Schedule(2024)
### Model Construction
* 6.24-6.25: Theory knowledge learning & understand used model
* 6.25-6.28: Construct models
* 7.15-7.17: Construct models
  * 7.15: processing(kmer)
  * 7.16: alter DeepAAI & graph generator
  * 7.17: combine the model

* 7.18: train code
* 7.22: train

### Experiment

...

### Conclusion

...

### ...


****

## Analysis to Basic Models & Thinking

### DeepAAI

#### Basics

* DeepAII combine local feature and global feature
  * loacl: CNN, extract feature from amino feature(processed)
  * global: GCN, extract feature from graph feature
    * combine multiple inputs(based on graph structure)
    * generate adjacency matrix
    * use GCN to generate by inputing node_ft and adj 
    * output: combine node_ft in different stages

#### Thinkings

* About local feature: 
  * CNN is too simple, maybe Transformer can utilize more info 
    * 7.15--> cur: keep using CNN to keep the model simple
  * Design with antibody's bio feature 
* About global feature:
  * GCN can be replaced to GAT
  * inputs 
    * 7.15--> cur: only use kmer
  * contrastive learning: GCN and GAT


### MCGCL-DTA

#### Basic

* genarate affinity graph
* use GCN and GAT to process the graph
* contrastive learning
