'''
Example of using HGAT for link prediction task on disease dataset
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import hypercore.nn as hnn
from hypercore.manifolds import PoincareBall
from hypercore.models import LPModel
import argparse
import numpy as np
import time
from hypercore.datasets import Dataset
from hypercore.optimizers import Optimizer, LR_Scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='disease_lp')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--model', type=str, default='HGAT')
parser.add_argument('--manifold', type=str, default='PoincareBall')
parser.add_argument('--eval-freq', type=int, default='10')
parser.add_argument('--cuda', type=int, default='-1')
args = parser.parse_args()
args.min_epoch = 1000
args.patience = 1000
print('file starts')

class HGAT(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim):
        super(HGAT, self).__init__()
        self.manifold = manifold
        self.conv1 = hnn.HGATConv(self.manifold, in_dim, hidden_dim, dropout=0.0, act=F.relu, use_bias=1, use_att=True)
        self.conv2 = hnn.HGATConv(self.manifold, hidden_dim, hidden_dim, dropout=0.0, act=F.relu, use_bias=1, use_att=True)

        # sets the curvature of the output of the encoder
        self.c = manifold.c
    def encode(self, x, adj):
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x)))
        x, adj = self.conv1((x, adj))
        y, adj = self.conv2((x, adj))
        return y
    
dataset = Dataset()
dataset.set_args(normalize_feats=0)
dataset.load_dataset(dataset=args.dataset, task='lp')
data = dataset.data
print('data_loaded')


num_nodes, feat_dim = data['features'].shape
nb_false_edges = len(data['train_edges_false'])
nb_edges = len(data['train_edges'])

manifold = PoincareBall(1.0)
model = LPModel(HGAT(manifold, feat_dim, 16), manifold, nb_false_edges, nb_edges, max_norm=5e3)
print('model created')
print(str(model))


tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f"Total number of parameters: {tot_params}")


optimizer = Optimizer(model, euc_lr=0.01)
print('optimizer made')
print(optimizer.optimizer)
lr_scheduler = LR_Scheduler(optimizer.optimizer[0], euc_gamma=0.5)

def loss_function(pos_scores, neg_scores):
    loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
    loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
    return loss

best_val_metrics = model.init_metric_dict()

for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train', loss_function)
        train_metrics['loss'].backward()
        max_norm = 0.5
        all_params = list(model.parameters())
        for param in all_params:
            torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val', loss_function)
                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = model.compute_metrics(embeddings, data, 'test', loss_function)
                    best_emb = embeddings.cpu()
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        break
                print('Epoch {}: AUCROC Score: {}'.format(epoch + 1, str(val_metrics['roc'])))
if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
print('Final: Best AUCROC Score: {}'.format(str(best_test_metrics['roc'])))