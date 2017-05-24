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
    sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(0,10000))),
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

        self.ls1 = nn.Linear(3*input_shape, input_shape)

        self.l1 = nn.Linear(3*input_shape, input_shape)

        self.sig = nn.Sigmoid()

    def forward(self, u, z):
        x = torch.cat((u, z, u*z), dim=1)
        s = self.sig(self.ls1(x))
        return s + self.l1(x)


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


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.e1 = nn.Linear(784, 300)
        self.e2 = nn.Linear(300, 20)
        self.d2 = nn.Linear(20, 300)
        self.d1 = nn.Linear(300, 784)

        self.act = nn.Sigmoid()

    def encode(self, x):
        x = self.e1(x)
        x = self.act(x)
        x = self.e2(x)
        x = self.act(x)
        return x

    def decode(self, z):
        z = self.d2(z)
        z = self.act(z)
        z = self.d1(z)
        z = self.act(z)
        return z

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        x = self.decode(z)
        return x





class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(std.size()), requires_grad=False)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = LN()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def ae_loss_function(recon_x, x):
    return reconstruction_function(recon_x, x)

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


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

    uns *= 1/len(recons)

    return sup + uns, sup, uns


optimizer = optim.Adam(model.parameters(), lr=0.002)

dae_weights = [1000, 10, 0.1, 0.1]


import torchsample

trainer = torchsample.modules.ModuleTrainer(None)

callbacks = [
    trainer.history,
    torchsample.callbacks.TQDM(),
]


def run_callbacks(method, *args, **kwargs):
    [getattr(c,method)(*args, **kwargs) for c in callbacks]

def train(epoch, epoch_logs, metrics):
    model.train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()

        batch_logs = {
            'batch_idx': batch_idx,
            'batch_samples': args.batch_size,
        }
        run_callbacks("on_batch_begin", batch_idx, batch_logs)

        y_pred, (refs, recs) = model(data)
        loss, sl, ul = ln_loss_function(recs, refs, dae_weights, y_pred, label)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        """
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            print(sl, ul)
        """

        batch_logs['loss'] = loss.data[0]
        epoch_logs['loss'] = loss.data[0]

        batch_logs.update(metrics(y_pred, label))

        run_callbacks("on_batch_end", batch_idx, batch_logs)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def validate(val_metrics):
    total_loss = 0
    for batch_idx, (data, label) in enumerate(val_loader):
        data, label = Variable(data), Variable(label)
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        y_pred, (refs, recs) = model(data)

        val_loss, *_ = ln_loss_function(recs, refs, dae_weights, y_pred, label)
        total_loss += val_loss

    return total_loss / float(len(val_loader) * args.batch_size), val_metrics(y_pred, label)

def test(epoch):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        data = Variable(data, volatile=True)
        recon_batch = model(data)
        test_loss += ln_loss_function(recon_batch, data).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


run_callbacks("set_model", trainer)
run_callbacks("on_train_begin", {'has_validation_data': False})

trainer.add_metric('accuracy')

metrics = torchsample.metrics.MetricsModule(trainer._metrics)
val_metrics = torchsample.metrics.MetricsModule(trainer._metrics, prefix='val_')

for epoch in range(1, args.epochs + 1):
    epoch_logs = {'nb_epoch': args.epochs, 'nb_batches': len(train_loader), 'has_validation_data': False}

    run_callbacks("on_epoch_begin", epoch-1, epoch_logs)
    train(epoch, epoch_logs, metrics)
    #test(epoch)

    val_loss, val_metric_logs = validate(val_metrics)

    print(trainer.history.batch_metrics.items())

    trainer.history.batch_metrics['val_loss'] = val_loss

    epoch_logs.update(trainer.history.batch_metrics)
    epoch_logs.update({k.split('_metric')[0]:v for k,v in val_metric_logs.items()})


    run_callbacks("on_epoch_end", epoch-1, epoch_logs)
