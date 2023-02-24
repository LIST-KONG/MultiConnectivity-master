import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from train import train_model
from eval import eval_FCSC
from load_data import FSDataset, FSDataset_GT
from divide import kfold_split, K_Fold, setup_seed
from multimodal.GCN import GCN as GCN_multi


parser = argparse.ArgumentParser(description='FC and SC Classification')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions (default: 10)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--threshold', type=float, default=0.2, help='threshold')
parser.add_argument('--sc_features', type=int, default=90, help='sc_features')
parser.add_argument('--fc_features', type=int, default=21, help='fc_features')
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes (HC/MDD)')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--num_layers', type=int, default=2, help='the numbers of convolution layers')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=400, help='patience for early stopping')
parser.add_argument('--dataset', type=str, default='multi', help="XX_SCFC/ZD_SCFC/HCP_SCFC")
parser.add_argument('--path', type=str, default=r'C:\Users\Administrator\Desktop\new_data_liufuyuan\new_data_liufuyuan\multi', help='path of dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--temperature', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--negative_weight', type=float, default=0.8, help='dropout ratio')
parser.add_argument('--num_atom_type', type=int, default=90, help='value for num_atom_type')
parser.add_argument('--num_edge_type', type=int, default=90, help='value for num_edge_type')
parser.add_argument('--num_heads', type=int, default=8, help='value for num_heads')
parser.add_argument('--in_feat_dropout', type=float, default=0.5, help='value for in_feat_dropout')
parser.add_argument('--readout', type=str, default='mean', help="mean/sum/max")
parser.add_argument('--layer_norm', type=bool, default=True, help="Please give a value for layer_norm")
parser.add_argument('--batch_norm', type=bool, default=False, help="Please give a value for batch_norm")
parser.add_argument('--residual', type=bool, default=True, help="Please give a value for residual")
parser.add_argument('--pos_enc_dim', type=int, default=32, help='hidden size')
parser.add_argument('--lap_pos_enc', type=bool, default=True, help="Please give a value for lap_pos_enc")
parser.add_argument('--wl_pos_enc', type=bool, default=False, help="Please give a value for wl_pos_enc")
args = parser.parse_args()


if __name__ == '__main__':
    acc = []
    loss = []
    sen = []
    spe = []
    f1 = []
    auc = []
    setup_seed(args.seed)
    random_s = np.array([175, 200, 225, 250, 275], dtype=int)  #25, 50, 100, 125, 150,
    # random_s = np.array([125, 150, 175, 200, 225, 250, 300], dtype=int)
    print(args)
    for k in range(5):
        print(k)
        # myDataset = FSDataset_GT(args)
        myDataset = FSDataset(args)
        myDataset.k_fold_split = K_Fold(args.repetitions,myDataset.choose_data, random_s[k])
        acc_iter = []
        loss_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []
        for i in range(10):
            print(i)
            train_loader, val_loader, test_loader = kfold_split(myDataset, i, args)

            # Model initialization
            # model = GraphTransformer(args).to(args.device)
            model = GCN_multi(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # Model training
            best_model = train_model(args, model, optimizer, train_loader, val_loader, test_loader, i)

            # Restore model for testing
            model.load_state_dict(torch.load('ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i)))
            test_acc, test_loss, test_sen, test_spe, test_f1, test_auc,  y, pred = eval_FCSC(args, model, test_loader)
            acc_iter.append(test_acc)
            loss_iter.append(test_loss)
            sen_iter.append(test_sen)
            spe_iter.append(test_spe)
            f1_iter.append(test_f1)
            auc_iter.append(test_auc)
            print('Test set results, best_epoch = {:.1f}  loss = {:.6f}, accuracy = {:.6f}, sensitivity = {:.6f}, '
                  'specificity = {:.6f}, f1_score = {:.6f}, auc_score = {:.6f}'.format(best_model, test_loss, test_acc, test_sen, test_spe, test_f1, test_auc))
            print(y)
            print(pred)
        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))
        f1.append(np.mean(f1_iter))
        auc.append(np.mean(auc_iter))

        print('Average test set results, mean accuracy = {:.6f}, mean_sen = {:.6f}, '
              'mean_spe = {:.6f}, mean_f1 = {:.6f}, mean_auc = {:.6f}'.format(acc[k], sen[k], spe[k], f1[k], auc[k]))
    print(args)
    print('Total test set results, accuracy : {}'.format(acc))
    print('Average test set results, mean accuracy = {:.6f}, std = {:.6f}, mean_sen = {:.6f}, std_sen = {:.6f}, '
          'mean_spe = {:.6f}, std_spe = {:.6f}, mean_f1 = {:.6f}, std_f1 = {:.6f}, mean_auc = {:.6f}, std_auc = {:.6f}'.format(np.mean(acc), np.std(acc), np.mean(sen), np.std(sen),
                                                       np.mean(spe), np.std(spe), np.mean(f1), np.std(f1), np.mean(auc), np.std(auc)))
