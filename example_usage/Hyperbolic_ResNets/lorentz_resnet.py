'''
Example of using a Lorentzian ResNet for image classification task on CIFAR-10

This example also shows how to use curvature that varies by layer
'''

from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import hypercore.nn as hnn
from hypercore.manifolds import Lorentz
import torch.nn.functional as F
from hypercore.optimizers import RiemannianSGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from hypercore.optimizers import Optimizer

# Define basic ResNet block for the Lorentzian ResNet using hyperbolic convolutional layer, batch normalziation, activation, and residual connection
class LorentzBasicBlock(nn.Module):
    def __init__(self, manifold, 
            in_channels, out_channels, stride=1):
        super(LorentzBasicBlock, self).__init__()
        self.manifold = manifold
        self.c = manifold.c
        self.conv1 = hnn.LorentzConv2d(manifold, 
                    in_channels, out_channels, 
                    kernel_size=3, stride=stride, padding=1)
        self.bn1 = hnn.LorentzBatchNorm2d(manifold, out_channels + 1) # plus 1 for time dimension
        self.act = hnn.LorentzActivation(manifold, nn.ReLU(inplace=True))
        self.conv2 = hnn.LorentzConv2d(manifold, 
                    out_channels + 1, out_channels, 
                    kernel_size=3, stride=1, padding=1) # plus 1 for time dimension
        self.bn2 = hnn.LorentzBatchNorm2d(manifold, out_channels + 1) # plus 1 for time dimension
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                hnn.LorentzConv2d(manifold, in_channels, out_channels, 
                                  kernel_size=1, stride=stride),
                hnn.LorentzBatchNorm2d(manifold, out_channels + 1) # plus 1 for time dimension
            )

        self.residual = hnn.LResNet(manifold, weight=0.5, scale=2.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.residual(out, self.shortcut(x))
        out = self.act(out)
        return out
    

    
class LorentzResNet18(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, num_classes=10):
        super(LorentzResNet18, self).__init__()
        self.in_channels = 64 + 1 # pluts 1 for time dimension
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.conv1 = hnn.LorentzConv2d(self.manifold_in, in_channels=3 + 1, out_channels=64, 
                                       kernel_size=3, stride=1, padding=1, manifold_out=self.manifold_hidden)
        self.bn1 = hnn.LorentzBatchNorm2d(self.manifold_hidden, 64 + 1) # plus 1 for time dimension
        self.act = hnn.LorentzActivation(self.manifold_hidden, F.relu) 
        
        self.layer1 = self._make_layer(LorentzBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(LorentzBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(LorentzBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(LorentzBasicBlock, 512, 2, stride=2)
        self.avg_pool = hnn.LorentzGlobalAvgPool2d(manifold_in=self.manifold_hidden, keep_dim=True, manifold_out=self.manifold_out)
    
        self.fc = hnn.LorentzMLR(self.manifold_out, 512 + 1, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.manifold_hidden, self.in_channels, 
                                out_channels, stride))
            self.in_channels = out_channels + 1 # plus 1 for time dimension
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) 
        x = self.manifold_in.projx(F.pad(x, pad=(1, 0)))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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

    manifold_in = Lorentz(0.9)
    manifold_hidden = Lorentz(1.0)
    manifold_out = Lorentz(1.1)
    model = LorentzResNet18(manifold_in, manifold_hidden, manifold_out)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Optimizer(model, euc_optimizer_type='sgd', euc_lr=0.1, euc_weight_decay=5e-4, hyp_optimizer_type='rsgd', hyp_lr=0.1, 
                          hyp_weight_decay=0, momentum_hyp=0.7, nesterov_euc=True, nesterov_hyp=True, stabilize=1)
    # RiemannianSGD(model.parameters(), lr=1e-1, weight_decay=1e-3, momentum=0.9, nesterov=True, stabilize=1)
    lr_scheduler_euc = MultiStepLR(optimizer.optimizer[0], milestones=[60,120,160], gamma=0.2)
    lr_scheduler_hyp = MultiStepLR(optimizer.optimizer[1], milestones=[120,160], gamma=0.2)

    for epoch in range(0, 200):
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
            lr_scheduler_euc.step()
            lr_scheduler_hyp.step()
            print("Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(epoch + 1, 200, np.mean(losses), np.mean(acc1), np.mean(acc5)))

    model_path = 'resnet18.pt'
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