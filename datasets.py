# +
import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'training-data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        if name == 'REDDIT-BINARY':
            dataset.transform = T.Constant()
        else:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            dataset.transform = T.OneHotDegree(max_degree)

    return dataset
# -


