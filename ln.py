from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--train-samples', type=int, default=50000)
parser.add_argument('--validation-samples', type=int, default=10000)
parser.add_argument('--noise-sigma', type=float, default=0.3)
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--loss-torch-mse', action='store_true',
                    help='use torch builtin MSE for recon loss instead of manual comp. to avoid need for detach')
parser.add_argument('--loss-normalization', type=str, choices=['ladder','mean','none'], default='ladder')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--parallel-encoder', action='store_true',
                    help='Execute noisy and clean encoder in parallel (default: off)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(0,args.train_samples))),
    batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(args.train_samples,args.train_samples+args.validation_samples))), **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class LinearCombinator(nn.Module):
    def __init__(self, input_shape):
        super(LinearCombinator, self).__init__()

        self.l1 = nn.Linear(input_shape, input_shape)
        self.l2 = nn.Linear(input_shape, input_shape)

    def forward(self, x_recon, x_short):
        return self.l1(x_recon) + self.l2(x_short)

class MLPCombinator(nn.Module):
    def __init__(self, input_shape):
        super(MLPCombinator, self).__init__()

        self.l1 = nn.Linear(3*input_shape, input_shape)
        self.l2 = nn.Linear(4*input_shape, input_shape)

        self.sig = nn.Sigmoid()

    def forward(self, u, z):
        x = torch.cat((u, z, u*z), dim=1)
        s = self.sig(self.l1(x))
        return self.l2(torch.cat((x,s), dim=1))


class Encoder(nn.Module):
    def __init__(self, layers, acts, noise_std=0.):
        super().__init__()
        self.acts = acts
        self.layers = layers
        self.noise_std = noise_std

    def noise(self, x):
        noise = Variable(torch.randn(x.size()) * self.noise_std)
        if args.cuda:
            noise = noise.cuda()
        return x + noise

    def encode(self, layer, act, x):
        if self.noise_std > 0:
            y = self.noise(layer(x))
        else:
            y = layer(x)
        y_act = act(y)
        return y, y_act

    def forward(self, e0):
        if self.noise_std > 0:
            e0 = self.noise(e0)

        e_act = e0
        pres = [e0]
        acts = []

        for (layer,act) in zip(self.layers, self.acts):
            e_pre, e_act = self.encode(layer, act, e_act)
            pres += [e_pre]
            acts += [e_act]

        return pres, acts

class LN(nn.Module):
    def __init__(self, parallel_encoder=False):
        super(LN, self).__init__()

        self.parallel_encoder = parallel_encoder

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

        self.e1 = nn.Linear(784, 300)
        self.e2 = nn.Linear(300, 20)
        self.e3 = nn.Linear(20, 10)

        self.d3 = nn.Linear(10, 20)
        self.d2 = nn.Linear(20, 300)
        self.d1 = nn.Linear(300, 784)

        self.g3 = MLPCombinator(10)
        self.g2 = MLPCombinator(20)
        self.g1 = MLPCombinator(300)
        self.g0 = MLPCombinator(784)

        acts = [self.relu, self.relu, self.softmax]
        layers = [self.e1, self.e2, self.e3]

        self.e_noisy = Encoder(layers=layers, acts=acts, noise_std=args.noise_sigma)
        self.e_clean = Encoder(layers=layers, acts=acts, noise_std=0.0)

        if self.parallel_encoder:
            self.e_noisy = torch.nn.DataParallel(self.e_noisy, device_ids=[0])
            self.e_clean = torch.nn.DataParallel(self.e_clean, device_ids=[0])


    def forward(self, x):
        x = x.view(-1, 784)

        (e0_clean,e1_clean,e2_clean,e3_clean), (e1_clean_act,e2_clean_act,e3_clean_act) = self.e_clean(x)
        (e0_noisy,e1_noisy,e2_noisy,e3_noisy), (e1_noisy_act,e2_noisy_act,e3_noisy_act) = self.e_noisy(x)

        l3_recon = self.g3(e3_noisy_act, e3_noisy)
        u2 = self.d3(l3_recon)
        l2_recon = self.g2(u2, e2_noisy)
        u1 = self.d2(l2_recon)
        l1_recon = self.g1(u1, e1_noisy)
        u0 = self.d1(l1_recon)
        l0_recon = self.g0(u0, e0_noisy)

        refs = [e0_clean, e1_clean, e2_clean, e3_clean]
        recs = [l0_recon, l1_recon, l2_recon, l3_recon]
        sup_noisy = e3_noisy_act
        sup_clean = e3_clean_act

        return sup_noisy, sup_clean, (refs, recs)


model = LN(parallel_encoder=args.parallel_encoder)
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


nll = nn.NLLLoss()
mse = nn.MSELoss()


def ln_loss_function(recons, refs, weights, y_pred, y_true):
    sup = nll(y_pred, y_true)
    uns = 0

    for i, _ in enumerate(recons):
        if args.loss_normalization == 'ladder':
            mean = refs[i].mean(dim=1).expand(refs[i].size())
            std = refs[i].std(dim=1).expand(refs[i].size())
            refs[i] = (refs[i] - mean) / std
            recons[i] = (recons[i] - mean) / std
        elif args.loss_normalization == 'mean':
            mean = refs[i].mean(dim=1).expand(refs[i].size())
            refs[i] = (refs[i] - mean)
            recons[i] = (recons[i] - mean)
        elif args.loss_normalization == 'none':
            pass

        if args.loss_torch_mse:
            uns += mse(recons[i], refs[i].detach()) * weights[i]
        else:
            uns += torch.mean((recons[i] - refs[i])**2) * weights[i]

    return sup + uns, sup, uns


optimizer = optim.Adam(model.parameters(), lr=0.002)

dae_weights = [1000, 10, 0.1, 0.1]


def train(epoch):
    model.train()
    train_loss = 0

    import time

    t0 = time.time()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()

        y_pred_noisy,  y_pred_clean, (refs, recs) = model(data)
        loss, sl, ul = ln_loss_function(recs, refs, dae_weights, y_pred_noisy, label)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            from scipy.misc import imsave
            imsave('refs_0_0.png', refs[0][0].cpu().data.numpy().reshape((28,28)))
            imsave('recs_0_0.png', recs[0][0].cpu().data.numpy().reshape((28,28)))

            print('Train Epoch: {epoch} [{bidx:8}/{batches:8} ({bperc:3.0f}%)]\tLoss: {loss:.6f} sup: {sup:.6f} dae: {dae:.6f} acc: {acc:.2f}'.format(**{
                'epoch': epoch,
                'bidx': batch_idx * len(data),
                'batches': len(train_loader.dataset),
                'bperc': 100. * batch_idx / len(train_loader),
                'loss': loss.data[0] / len(data),
                'sup': sl.data[0],
                'dae': ul.data[0],
                'acc': y_pred_noisy.data.max(1)[1].eq(label.data).cpu().sum() / len(data),
            }))
    tt = time.time()

    print('====> Epoch: {} Average loss: {:.4f} Time: {:.2}s'.format(
          epoch, train_loss / len(train_loader.dataset), tt-t0))

def validate(epoch):
    model.eval()

    total_loss = 0
    correct = 0
    n = 0

    for batch_idx, (data, label) in enumerate(val_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        y_pred_noisy, y_pred_clean, (refs, recs) = model(data)

        val_loss, *_ = ln_loss_function(recs, refs, dae_weights, y_pred_clean, label)
        total_loss += val_loss.data[0]

        pred = y_pred_clean.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(label.data).cpu().sum()
        n += data.size(0)

    total_loss /= float(len(val_loader) * args.batch_size)
    print('====> Validation set loss: {total_loss:.4f}, Acc.: {acc:.2f}/{n} ({accperc:.2f})'.format(**{
        'total_loss':total_loss,
        'acc': correct,
        'n': n,
        'accperc': correct / n,
    }))

def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        data = Variable(data, volatile=True)
        recon_batch = model(data)
        test_loss += ln_loss_function(recon_batch, data).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    #test(epoch)
