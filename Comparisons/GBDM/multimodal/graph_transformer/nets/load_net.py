"""
    Utility file to select GraphNN model as
    selected by the user
"""
from multimodal.graph_transformer.nets.graph_transformer_net import GraphTransformerNet

def GraphTransformer(args):
    return GraphTransformerNet(args)

def gnn_model(args):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models(args)