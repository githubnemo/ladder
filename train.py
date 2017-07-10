from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

from logger import Logger
from dotter import make_dot
from ln import LN

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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batchnorm', choices=['off','decoder','encoder','all'], default='off')
parser.add_argument('--name', default='', type=str)

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


model = LN(batchnorm_mode=args.batchnorm, cuda=args.cuda, noise_std=args.noise_sigma)
if args.cuda:
    model.cuda()


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

log_path = os.path.join('logs', args.name)
logger = Logger(log_path, running_naming=not args.name)

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
            #from scipy.misc import imsave
            #imsave('refs_0_0.png', refs[0][0].cpu().data.numpy().reshape((28,28)))
            #imsave('recs_0_0.png', recs[0][0].cpu().data.numpy().reshape((28,28)))

            acc = y_pred_noisy.data.max(1)[1].eq(label.data).cpu().sum() / len(data)
            step = (epoch-1) * len(train_loader) + batch_idx
            sup_loss = sl.data[0]
            uns_loss = ul.data[0]

            logger.scalar_summary('loss', loss.data[0], step)
            logger.scalar_summary('acc', acc, step)
            logger.scalar_summary('sup_loss', sup_loss, step)
            logger.scalar_summary('dae_loss', uns_loss, step)
            logger.image_summary('recs', recs[0][0].cpu().view(-1,28,28).data.numpy(), step)

            def to_np(x): return x.data.cpu().numpy()
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), step+1)
                logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

            print('Train Epoch: {epoch} [{bidx:8}/{batches:8} ({bperc:3.0f}%)]\tLoss: {loss:.6f} sup: {sup:.6f} dae: {dae:.6f} acc: {acc:.2f}'.format(**{
                'epoch': epoch,
                'bidx': batch_idx * len(data),
                'batches': len(train_loader.dataset),
                'bperc': 100. * batch_idx / len(train_loader),
                'loss': loss.data[0] / len(data),
                'sup': sup_loss,
                'dae': uns_loss,
                'acc': acc,
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
    
    logger.scalar_summary('val_loss', total_loss.data[0], step)

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

with open(os.path.join(logger.run_path, 'model.pkl'),'wb') as f:
    torch.save(model, f)
