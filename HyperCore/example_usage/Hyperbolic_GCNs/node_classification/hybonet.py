'''
Example of using skip-connected HyboNet for noce classification task on disease dataset
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import hypercore.nn as hnn
from hypercore.manifolds import Lorentz
from hypercore.models import NCModel
import argparse
import numpy as np
import time
from hypercore.datasets import Dataset
from hypercore.models.graph_decoders import LorentzDecoder
from hypercore.optimizers import Optimizer, LR_Scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='disease_nc')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--model', type=str, default='HyboNet')
parser.add_argument('--eval-freq', type=int, default='10')
parser.add_argument('--cuda', type=int, default='-1')
args = parser.parse_args()
args.min_epoch = 1000
args.patience = 500
print('file starts')

class HyboNet(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim):
        super(HyboNet, self).__init__()
        
        self.manifold = manifold
        self.conv1 = hnn.HybonetConv(self.manifold, in_dim, hidden_dim, use_bias=True, dropout=0.1, use_att=False, local_agg=False, nonlin=None)
        self.conv2 = hnn.HybonetConv(self.manifold, hidden_dim, hidden_dim, use_bias=True, dropout=0.1, use_att=False, local_agg=False, nonlin=F.relu)
        self.conv3 = hnn.HybonetConv(self.manifold, hidden_dim, hidden_dim, use_bias=True, dropout=0.1, use_att=False, local_agg=False, nonlin=F.relu)
        self.conv4 = hnn.HybonetConv(self.manifold, hidden_dim, hidden_dim, use_bias=True, dropout=0.1, use_att=False, local_agg=False, nonlin=F.relu)
        self.residual = hnn.LResNet(self.manifold)

        self.decoder = LorentzDecoder(self.manifold, hidden_dim, out_dim, use_bias=True)

        self.c = manifold.c

    def encode(self, x, adj):
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x)))
        x0 = x
        x1, adj = self.conv1((x0, adj))
        x2, adj = self.conv2((x1, adj))
        x2 = self.residual(x1, x2)
        x3, adj = self.conv3((x2, adj))
        x3 = self.residual(x2, x3)
        x4, adj = self.conv4((x3, adj))
        return  self.residual(x3, x4)
    
args.margin = 2.
args.bias = 1

dataset = Dataset()
dataset.load_dataset(dataset=args.dataset, task='nc')
data = dataset.data
print('data_loaded')


n_classes = int(data['labels'].max() + 1)
n_nodes, feat_dim = data['features'].shape

manifold = Lorentz(c=1.0, learnable=False)
encoder = HyboNet(manifold, feat_dim + 1, 16, n_classes)
decoder = encoder.decoder
model = NCModel(encoder, decoder)
print('model created')
print(str(model))


tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f"Total number of parameters: {tot_params}")


optimizer = Optimizer(model, euc_lr=0.005, hyp_lr=0.005)
print('optimizer made')
print(optimizer.optimizer)
# lr_scheduler = LR_Scheduler(optimizer.optimizer[0], euc_gamma=0.5, optimizer_hyp=optimizer.optimizer[1], hyp_gamma=0.5)


def loss_function(output, data_input, weights):
    correct = output.gather(1, data_input.unsqueeze(-1))
    loss = F.relu(args.margin - correct + output).mean()
    return loss

best_val_metrics = model.init_metric_dict()
best_test_metrics = None

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
        # lr_scheduler.step()
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
                print('Epoch{}: f1 Score: {}'.format(epoch + 1, str(val_metrics['f1'])))
if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
print('Final: Best f1 Score: {}'.format(str(best_test_metrics['f1'])))
