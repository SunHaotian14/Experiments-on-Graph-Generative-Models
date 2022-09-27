from lib2to3.pytree import Node
import pickle
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import numpy as np

# load a list of graphs
def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    for i in range(len(graph_list)):
        edges_with_selfloops = graph_list[i].selfloop_edges()
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list

class CommunityDataset(Dataset):
     def __init__(self, adj_features_list, node_features_list):
         super(Dataset, self).__init__()
         self.x = node_features_list
         self.adj_features = adj_features_list
        
     def __len__(self):
         return len(adj_features_list)
        
     def __getitem__(self, index):
        return adj_features_list[index], node_features_list[index]

dataset_path = 'GraphRNN/graphs/GraphRNN_RNN_caveman_small_4_64_train_0.dat'
graph_list = load_graph_list(dataset_path)
adj_features_list = []
node_features_list = []
max_node_num = max([g.number_of_nodes() for g in graph_list])

degrees = [list(g.degree().values()) for g in graph_list]
degree_list = list(set([item for sublist in degrees for item in sublist]))
print(degree_list)

for g in graph_list:
    adj_tensor = torch.zeros((2, max_node_num, max_node_num))
    adj_tensor[0, :g.number_of_nodes(), :g.number_of_nodes()] = torch.tensor(nx.adjacency_matrix(g).toarray())
    adj_tensor[1, :g.number_of_nodes(), :g.number_of_nodes()] = 1 - torch.tensor(nx.adjacency_matrix(g).toarray())
    # one-hot encoding
    node_degree = np.array(list(g.degree().values()), dtype=int) - 1
    #(B, N, node_dim)
    node_tensor = torch.zeros(max_node_num, len(degree_list))
    for idx, deg in enumerate(node_degree):
        node_tensor[idx, degree_list == deg] = 1

    #(B, 1, N, N)
    # adj_tensor = adj_tensor.unsqueeze(0)
    adj_features_list.append(adj_tensor)
    node_features_list.append(node_tensor)

data_train = CommunityDataset(adj_features_list, node_features_list)

data_train_loader = DataLoader(data_train, batch_size=16, shuffle=False)
for batch_idx, data_batch in enumerate(data_train_loader):
    if batch_idx == 0:
        adj_features, node_features = data_batch 
        #(B, N, node_dim)
        print(node_features.shape)
        #(B, 2, N, N)
        print(adj_features.shape)

from dig.ggraph.method import GraphDF, GraphAF
import json
from dig.ggraph.dataset import ZINC250k
from torch_geometric.loader import DenseDataLoader
import numpy as np

runner = GraphAF()

config_dict =  {
        "max_size": 20,
        "edge_unroll": 12,
        "node_dim": 10,
        "bond_dim": 2,
        "num_flow_layer": 12,
        "num_rgcn_layer": 3,
        "nhid": 128,
        "nout": 128,
        "deq_coeff": 0.9,
        "st_type": "exp",
        "use_gpu": True,
        "use_df": False
    }

torch.cuda.empty_cache()

runner.train_rand_gen(loader=data_train_loader, lr = 0.001, wd = 0, max_epochs = 1000, 
                        model_conf_dict = config_dict, save_interval = 1, save_dir = 'community_test')