import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score

def sensitivity_specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    # print("tn:", tn, "tp:", tp, "fn:", fn, "fp:", fp)
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)

    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity, specificity




def eval_FCSC(args, model, loader):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for data in loader:
        with torch.no_grad():
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            test_loss += F.nll_loss(out, data.y).item()
            pred_num = pred.cpu().numpy()
            y_num = data.y.cpu().numpy()
            for num in range(len(pred)):
                Y_pred.append(pred_num[num])
                Y_test.append(y_num[num])
    test_acc = correct / len(loader.dataset)
    test_sen, test_spe = sensitivity_specificity(Y_test, Y_pred)
    test_f1=f1_score(Y_test, Y_pred)
    test_auc=roc_auc_score(Y_test, Y_pred)

    # return test_acc, test_loss, test_sen, test_spe, test_f1, Y_test, Y_pred
    return test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, Y_test, Y_pred