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
from ln import LN

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--train-samples', type=int, default=50000)
parser.add_argument('--validation-samples', type=int, default=10000)
parser.add_argument('--noise-sigma', type=float, default=0.3)
parser.add_argument('--samples-j', type=int, default=1, metavar='N')
parser.add_argument('--samples-k', type=int, default=1, metavar='N')
parser.add_argument('--class', dest='cls', type=int, default=2, metavar='N')
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
parser.add_argument('checkpoint', type=argparse.FileType('r'))
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


model = torch.load(args.checkpoint.name)
if args.cuda:
    model.cuda()

logger = Logger('./logs')



z = Variable(torch.FloatTensor([[0,0,0,0,0,0,0,0,0,0]]))
z.data[0][args.cls] = 1
if args.cuda:
    z = z.cuda()

model.e_noisy.noise_std = args.noise_sigma
#sample = model.sample_j(z, args.samples_k, args.samples_j)
#sample = model.sample_ble(z, k=args.samples_k, j=args.samples_j)
#sample = model.sample_s(z, k=args.samples_k, j=args.samples_j)
#print(sample)

#sample = sample.clamp(min=-255, max=255)
#from scipy.misc import imsave
#imsave('sample.png', sample.cpu().data.numpy().reshape((28,28)))


from scipy.misc import imsave

batch_data, batch_labels = list(train_loader)[0]
batch_data = Variable(batch_data)
if args.cuda:
    batch_data = batch_data.cuda()
    batch_labels = batch_labels.cuda()
example = batch_data[0].view(-1, 784)

print(example)

samples = model.sample_by_example(example, k=args.samples_k)

imsave('example_sample_example.png', example.cpu().data.numpy().reshape(28,28))

for i, sample in enumerate(samples):
    sample = sample.clamp(min=-255, max=255)
    imsave('example_sample_{}.png'.format(i+1), sample.cpu().data.numpy().reshape((28,28)))


