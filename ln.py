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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
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
    batch_size=args.batch_size, shuffle=True,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(0,50000))),
    **kwargs)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(50000,60000))), **kwargs)
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


class LN(nn.Module):
    def __init__(self):
        super(LN, self).__init__()

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

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

        self.noise_std = 0.3

    def noise(self, x):
        noise = Variable(torch.randn(x.size()) * self.noise_std)
        if args.cuda:
            noise = noise.cuda()
        return x + noise

    def encode(self, layer, act, x_clean, x_noisy):
        e_clean = layer(x_clean)
        e_noisy = self.noise(layer(x_noisy))
        e_clean_act = act(e_clean)
        e_noisy_act = act(e_noisy)
        return e_clean, e_noisy, e_clean_act, e_noisy_act

    def forward(self, x):
        e0_clean = x.view(-1, 784)
        e0_noisy = self.noise(x.view(-1,784))

        e1_clean, e1_noisy, e1_clean_act, e1_noisy_act = self.encode(self.e1, self.relu, e0_clean, e0_noisy)
        e2_clean, e2_noisy, e2_clean_act, e2_noisy_act = self.encode(self.e2, self.relu, e1_clean, e1_noisy)
        e3_clean, e3_noisy, e3_clean_act, e3_noisy_act = self.encode(self.e3, self.softmax, e2_clean, e2_noisy)

        l3_recon = self.g3(e3_noisy_act, e3_noisy)
        u2 = self.d3(l3_recon)
        l2_recon = self.g2(u2, e2_noisy)
        u1 = self.d2(l2_recon)
        l1_recon = self.g1(u1, e1_noisy)
        u0 = self.d1(l1_recon)
        l0_recon = self.g0(u0, e0_noisy)

        refs = [e0_clean, e1_clean, e2_clean, e3_clean]
        recs = [l0_recon, l1_recon, l2_recon, l3_recon]
        sup = e3_noisy_act

        return sup, (refs, recs)


model = LN()
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
        mean = refs[i].mean(dim=1).expand(refs[i].size())
        std = refs[i].std(dim=1).expand(refs[i].size())
        refs[i] = (refs[i] - mean) / std
        recons[i] = (recons[i] - mean) / std
        uns += mse(recons[i], refs[i].detach()) * weights[i]

    return sup + uns, sup, uns


optimizer = optim.Adam(model.parameters(), lr=0.002)

dae_weights = [1000, 10, 0.1, 0.1]


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()

        y_pred, (refs, recs) = model(data)
        loss, sl, ul = ln_loss_function(recs, refs, dae_weights, y_pred, label)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            print(sl, ul)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def validate(epoch):
    model.eval()

    total_loss = 0
    for batch_idx, (data, label) in enumerate(val_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        y_pred, (refs, recs) = model(data)

        val_loss, *_ = ln_loss_function(recs, refs, dae_weights, y_pred, label)
        total_loss += val_loss

    total_loss /= float(len(val_loader) * args.batch_size)
    print("====> Validation set loss: {:.4f}".format(total_loss))

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
