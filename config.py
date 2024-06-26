

## GATs --------------------------------------------
class GATConfig:
    dropout_rate = 0.2
    layer_dims = [512, 512, 256]


## GCNs --------------------------------------------
class GCNConfig:
    dropout_rate = 0.2
    layer_dims = [512, 512, 256]
    
    
## AttnAgg -----------------------------------------
class AttnAggConfig:
    def __init__(self, embedding_dim):
        self.dropout_rate = 0.2
        self.layer_dims = [256, embedding_dim, 1] 

## Contrastive Learning ----------------------------
class CLConfig:
    graph_init_dim = 2339
    embedding_dim = 256
    
    graph_dims = [256, 512, 256]
    feature_mlp_dims = [graph_init_dim, 512, 256]
    dr_mlp_dims = [128, 256, 128]
    pr_mlp_dims = [100, 256, 128]
    
    gcnConfig = GCNConfig()
    gatConfig = GATConfig()
    AttnAggConfig = AttnAggConfig(embedding_dim)
    


## Predictor ---------------------------------------
class PredictorConfig:
    dropout_rate = 0.1
    layer_dims = [512, 256, 128, 1]