## Abstract
* This repository is focusing on developing a model that can predict neutralization effect of antibody.
* based on GCNs, GATs, MLP

****
## Basics & Thinking

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
  * CNN is too simple, maybe Transformer can utilize more info.
  * Design with antibody's bio feature
* About global feature:
  * GCN can be replaced to GAT
  * inputs
  * contrastive learning: GCN and GAT


### 


****

## OurWork


## TODO
* constrct models
* generate graph
* train code
* predict code
* run & debug
* experiment

****

## Schedule(2024)
### Model Construction
* 6.24-6.25: Theory knowledge learning & understand used model
* 6.25-6.28: Construct models

### Experiment


### Conclusion


### ...


****

