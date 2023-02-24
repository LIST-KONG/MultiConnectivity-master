import argparse
import numpy as np
from sklearn.model_selection import KFold
import torch
import random
from torch.utils.data import Subset
from torch_geometric.data import DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices

def kfold_split(self, test_index, args):
    self.k_fold = args.repetitions
    self.batch_size = args.batch_size
    assert test_index < self.k_fold
    valid_index = test_index
    test_split = self.k_fold_split[test_index]
    valid_split = self.k_fold_split[valid_index]

    train_mask = np.ones(len(self.choose_data))
    train_mask[test_split] = 0
    train_mask[valid_split] = 0
    train_split = train_mask.nonzero()[0]

    train_subset = Subset(self.choose_data, train_split.tolist())
    valid_subset = Subset(self.choose_data, valid_split.tolist())
    test_subset = Subset(self.choose_data, test_split.tolist())

    train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 多中心MDD按中心划分数据集和测试集，有batch划分，真数据+假数据
def k_site_fold_batch(k, fold_id, real_data, fake_data, args):
    """
    :param k: 数据中心的个数，k=20表示一共有20个数据中心
    :param fold_id: 取第fold_id个中心数据作为测试集，其他均为训练集
    :param data: 输入所有中心数据
    :return: 返回训练集和测试集
    """
    train_data = []
    test_data = []
    for i in range(k):
        if (i == fold_id):
            test_data = real_data[i]
        else:
            train_data = train_data + real_data[i]
            train_data = train_data + fake_data[i]
    train_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader