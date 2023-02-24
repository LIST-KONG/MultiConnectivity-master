import time
import torch
import torch.nn.functional as F
from eval import eval_FCSC


def train_model(args, model, optimizer, train_loader, val_loader, test_loader, i_fold):
    """
    :param train_loader:
    :param model: model
    :type optimizer: Optimizer
    """
    min_loss = 1e10
    max_acc = 0
    patience = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            # loss = F.nll_loss(out, data.y)
            cross_loss = torch.nn.CrossEntropyLoss()
            loss = cross_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # val_acc, val_loss, val_sen, val_spe, val_f1,  _, _ = eval_FCSC(args, model, val_loader)
        # test_acc, test_loss, test_sen, test_spe, test_f1,  _, _ = eval_FCSC(args, model, test_loader)
        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = eval_FCSC(args, model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = eval_FCSC(args, model, test_loader)


        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
              'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))

        # if val_acc > max_acc:
        #     max_acc = val_acc
        #     torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
        #     print("Model saved at epoch{}".format(epoch))
        #     best_epoch = epoch
        #     patience = 0
        # elif val_acc == max_acc:
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
            print("Model saved at epoch{}".format(epoch))
            best_epoch = epoch
            min_loss = val_loss
            patience = 0
            # else:
            #    patience += 1
        else:
            patience += 1

        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))

    return best_epoch


