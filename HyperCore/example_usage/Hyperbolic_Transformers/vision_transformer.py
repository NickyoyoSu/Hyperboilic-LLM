'''
Example of building a hyperbolic vision transformer
'''
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import hypercore.nn as hnn
import torch.nn.functional as F
from hypercore.manifolds import Lorentz
from torch.optim.lr_scheduler import MultiStepLR
from hypercore.optimizers import Optimizer, LR_Scheduler
import numpy as np
import math
from geoopt import ManifoldParameter

class HyperbolicMLP(nn.Module):
    """
    A hyperbolic multi-layer perceptron module.
    """

    def __init__(self, manifold, in_channel, hidden_channel, dropout=0):
        super().__init__()
        self.manifold = manifold
        self.dense_1 = hnn.LorentzLinear(self.manifold, in_channel, hidden_channel)
        self.dense_2 = hnn.LorentzLinear(self.manifold, hidden_channel + 1, in_channel - 1)
        self.activation = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
class LorentzTransformerBlock(nn.Module):
    """
    A single lorentz transformer block.
    """

    def __init__(self, manifold, in_channel, hidden_channel, dropout=0, num_heads=1, output_attentions=False):
        super().__init__()
        self.attention = hnn.LorentzMultiheadAttention(manifold, in_channel, in_channel - 1, num_heads, attention_type='full', trans_heads_concat=True)
        self.layernorm_1 = hnn.LorentzLayerNorm(manifold, in_channel - 1)
        self.layernorm_2 = hnn.LorentzLayerNorm(manifold, in_channel - 1)
        self.mlp = HyperbolicMLP(manifold, in_channel, hidden_channel, dropout=dropout)
        self.output_attentions = output_attentions
        self.manifold = manifold
        self.residual_1 = hnn.LResNet(manifold, use_scale=True, scale=5.0) # for deep models, scale can be quite large e.g. ~30
        self.residual_2 = hnn.LResNet(manifold, use_scale=True, scale=5.0)

    def forward(self, x, output_attentions=False):
        # Self-attention
        x = self.layernorm_1(x)
        attention_output = self.attention(x, x, output_attentions=output_attentions)
        # # Skip connection
        x = self.residual_1(x, attention_output)
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = self.residual_2(x, mlp_output)
        # Return the transformer block's output and the attention probabilities (optional)
        return x
    
class LViTEncoder(nn.Module):
    """
    The lorentzian vision transformer encoder module.
    """

    def __init__(self, manifold_in, num_layers, in_channel, hidden_channel, num_heads=1, dropout=0, output_attentions=False, manifold_out=None):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            block = LorentzTransformerBlock(manifold_in, in_channel, hidden_channel, dropout, num_heads, output_attentions)
            self.blocks.append(block)
        self.fc = hnn.LorentzLinear(manifold_in, in_channel, in_channel - 1, manifold_out=manifold_out)
        self.manifold_out = manifold_out
        self.manifold = manifold_in

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        for block in self.blocks:
            x = block(x, output_attentions=output_attentions)
        return self.fc(x)
    



class LViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, 
                 manifold_in, 
                 manifold_hidden, 
                 manifold_out,
                 image_size=32, 
                 patch_size=4,
                 num_layers=6, 
                 in_channel=3, 
                 hidden_channel=64, #dimension per head
                 out_channel=10, 
                 mlp_hidden_size=64*4, 
                 num_heads=8, 
                 dropout=0.1, 
                 output_attentions=False):
        super().__init__()
        self.in_channel = in_channel + 1
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2
        # Create the embedding module
        self.patch_embedding = hnn.LorentzPatchEmbedding(manifold_in, image_size, patch_size, self.in_channel, self.hidden_channel * num_heads)
        self.positional_encoding = ManifoldParameter(self.manifold_in.random_normal((1, self.num_patches, self.hidden_channel + 1)), manifold=self.manifold_in, requires_grad=True)
        self.dropout = hnn.LorentzDropout(self.manifold_hidden, dropout=dropout)
        self.add_pos = hnn.LResNet(manifold_hidden, use_scale=True, scale=3.0)
        # Create the transformer encoder module
        self.encoder = LViTEncoder(self.manifold_hidden, self.num_layers, self.hidden_channel + 1, mlp_hidden_size, num_heads, dropout, output_attentions)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = hnn.LorentzMLR(self.manifold_out, self.hidden_channel + 1, self.out_channel)
        self.dropout = hnn.LorentzDropout(self.manifold_hidden, dropout=dropout)
        self.fc = hnn.LorentzLinear(manifold_in=manifold_in, manifold_out=self.manifold_hidden, in_features=self.hidden_channel + 1, out_features=self.hidden_channel)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        x = x.permute(0, 2, 3, 1) 
        x_hyp = self.manifold_in.projx(F.pad(x, pad=(1, 0)))
        embedding_output = self.patch_embedding(x_hyp)
        embedding_output = self.dropout(self.add_pos(embedding_output, self.positional_encoding))
        embedding_output = self.fc(embedding_output)
        # Calculate the encoder's output
        encoder_output = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits and return
        out = self.classifier(self.manifold_out.lorentzian_centroid(encoder_output))
        assert(not out.isnan().any())
        assert(not out.isinf().any())
        return out
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train():
    # import dataset
    train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
        ])

    test_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),
            ])

    train_set = datasets.CIFAR10('hypercore/data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10('hypercore/data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=8, pin_memory=True, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    manifold_in = Lorentz(1.0)
    manifold_hidden = Lorentz(0.75)
    manifold_out = Lorentz(0.75)
    model = LViT(manifold_in, manifold_hidden, manifold_out)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Optimizer(model, euc_optimizer_type='adamW', euc_lr=1e-4, euc_weight_decay=1e-2, hyp_optimizer_type='radam', hyp_lr=1e-4, 
                          hyp_weight_decay=1e-4, stabilize=1)
    print(optimizer.optimizer)
    print(str(model))
    lr_scheduler = LR_Scheduler(optimizer_euc=optimizer.optimizer[0], euc_lr_reduce_freq=30, euc_gamma=0.1, hyp_gamma=0.1, hyp_lr_reduce_freq=30, optimizer_hyp=optimizer.optimizer[1])
    for epoch in range(0, 300):
        model.train()
        losses = []
        acc1 = []
        acc5 = []

        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.append(loss.item())
                acc1.append(top1.item())
                acc5.append(top5.item())

        with torch.no_grad():
            lr_scheduler.step()
            print("Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(epoch + 1, 200, np.mean(losses), np.mean(acc1), np.mean(acc5)))

    model_path = 'Lorentz_ViT.pt'
    torch.save(model.state_dict(), model_path)

    print("Testing")
    model.eval()
    losses = []
    acc1 = []
    acc5 = []

    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.append(loss.item())
        acc1.append(top1.item(), x.shape[0])
        acc5.append(top5.item(), x.shape[0])

    loss_test = np.mean(losses)
    acc1_test = np.mean(acc1)
    acc5_test = np.mean(acc5)

    print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(loss_test, acc1_test, acc5_test))
if __name__ == '__main__':
    train()
