import os
import torch
from loaddatas import load_synthetic_data, get_disease_edgelist
import loaddatas as lds
import torch.nn.functional as F
import numpy as np
import SIMBLOCKGNN as SIMGNN
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.nn.init import xavier_normal_ as xavier
from torch_geometric.data import Data

path = os.getcwd()

def train():
    model.train()
    optimizer.zero_grad()
    emb = model.g_encode(data).clone()
    x, y = model.s_encode(data, emb) # emb from encode's, i.e., Gconv's output
    loss = F.binary_cross_entropy(x,y)
    loss.backward()
    optimizer.step()
    return x

def test():
    model.eval()
    accs = []
    emb = model.g_encode(data)
    for type in ["val", "test"]:
        pred,y = model.s_encode(data,emb,type=type)
        pred,y = pred.cpu(),y.cpu()
        if type == "val":
            accs.append(F.binary_cross_entropy(pred, y))
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y,pred)
            accs.append(acc)
        else:
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y, pred)
            accs.append(acc)
    return accs

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if not m.bias is None:
            torch.nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


d_names = ["Disease_M"]
times=range(1)


wait_total= 500
total_epochs = 2000


pipelines=['SIMGNN'] # where pipelines was defined!
pipeline_acc={'SIMGNN':[i for i in times]}
pipeline_acc_sum={'SIMGNN':0}
pipeline_roc={'SIMGNN':[i for i in times]}
pipeline_roc_sum={'SIMGNN':0}
pipeline_acc_same={'SIMGNN':[i for i in times]}
pipeline_acc_same_sum={'SIMGNN':0}
pipeline_roc_same={'SIMGNN':[i for i in times]}
pipeline_roc_same_sum={'SIMGNN':0}
pipeline_acc_diff={'SIMGNN':[i for i in times]}
pipeline_acc_diff_sum={'SIMGNN':0}
pipeline_roc_diff={'SIMGNN':[i for i in times]}
pipeline_roc_diff_sum={'SIMGNN':0}


for d_name in d_names:
    for data_cnt in times:
        for Conv_method in pipelines:
            if d_name in ['Rand_nnodes_github1000', 'PPI']:
                data = dataset[data_cnt]
            else:
                adj, features, y = load_synthetic_data('disease_lp', use_feats=True, data_path=path + "/disease_lp")
                dataset = Data(name='Disease_M', x = features, edge_index= get_disease_edgelist(adj), y = y, num_classes = None)
                data = dataset
            if d_name in ['Rand_nnodes_github1000']:
                data.x = data.x[:, :10]
            if d_name != "PPI":
                model, data = locals()[Conv_method].call(data, dataset.name, data.x.size(1), num_classes = None)
            else:
                model, data = locals()[Conv_method].call(data, 'PPI', data.x.size(1), dataset.num_classes)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0)
            best_val_acc = test_acc_same = test_acc_diff = test_acc = 0.0
            best_val_roc = test_roc_same = test_roc_diff = test_roc = 0.0
            best_val_loss = np.inf
            # train and val/test
            wait_step = 0

            # train and test
            for epoch in range(1, total_epochs + 1):
                pred = train()
                val_loss, val_roc, val_acc, tmp_test_roc, tmp_test_acc = test()
                if val_roc >= best_val_roc:
                    test_acc = tmp_test_acc
                    test_roc = tmp_test_roc
                    best_val_acc = val_acc
                    best_val_roc = val_roc
                    best_val_loss = val_loss
                    wait_step = 0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc,
                              ', Max roc: ', best_val_roc)
                        break
            del model
            del data

            pipeline_acc[Conv_method][data_cnt] = test_acc
            pipeline_roc[Conv_method][data_cnt] = test_roc

            log = 'Epoch: ' + str(
                total_epochs) + ', dataset name: ' + d_name + ', Method: ' + Conv_method + ' Test pr: {:.4f}, roc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][data_cnt], pipeline_roc[Conv_method][data_cnt])))
