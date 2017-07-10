from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable



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

        self.a1 = nn.Parameter(torch.Tensor(input_shape))
        self.a2 = nn.Parameter(torch.Tensor(input_shape))
        self.a3 = nn.Parameter(torch.Tensor(input_shape))
        self.a4 = nn.Parameter(torch.Tensor(input_shape))
        self.a5 = nn.Parameter(torch.Tensor(input_shape))
        self.b1 = nn.Parameter(torch.Tensor(input_shape))
        self.b2 = nn.Parameter(torch.Tensor(input_shape))
        self.b3 = nn.Parameter(torch.Tensor(input_shape))
        self.b4 = nn.Parameter(torch.Tensor(input_shape))

        self.a1.data.zero_()
        self.a2.data.zero_().add_(1)
        self.a3.data.zero_()
        self.a4.data.zero_().add_(1)
        self.a5.data.zero_()
        self.b1.data.zero_()
        self.b2.data.zero_().add_(1)
        self.b3.data.zero_()
        self.b4.data.zero_()

        self.sig = nn.Sigmoid()

    def forward(self, u, z_lat):
        a1 = self.a1.expand(u.size())
        a2 = self.a2.expand(u.size())
        a3 = self.a3.expand(u.size())
        a4 = self.a4.expand(u.size())
        a5 = self.a5.expand(u.size())
        b1 = self.b1.expand(u.size())
        b2 = self.b2.expand(u.size())
        b3 = self.b3.expand(u.size())
        b4 = self.b4.expand(u.size())

        s = self.sig(b1 * u + b2 * z_lat + b3 * u * z_lat + b4)
        return a1 * u + a2 * z_lat + a3 * u * z_lat + a4 * s + a5




class TwoStepBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(TwoStepBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = torch.nn.parameter.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                    .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        mean = input.mean(dim=1).expand(input.size())
        std = input.std(dim=1).expand(input.size())

        input = torch.nn.functional.batch_norm(
                input, self.running_mean, self.running_var, None, None,
                self.training, self.momentum, self.eps)

        return mean, std, input, self.weight, self.bias

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))



class LN(nn.Module):

    def __init__(self, batchnorm_mode='off', parallel_encoder=False, noise_std=0.3, cuda=False):
        super(LN, self).__init__()

        self.batchnorm_mode = []

        if batchnorm_mode == 'all':
            self.batchnorm_mode = ['encoder','decoder']
        elif batchnorm_mode == 'decoder':
            self.batchnorm_mode = ['decoder']
        elif batchnorm_mode == 'encoder':
            self.batchnorm_mode = ['encoder']

        self.parallel_encoder = parallel_encoder
        self.noise_std = noise_std

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

        self.e1 = nn.Linear(784, 300)
        self.e2 = nn.Linear(300, 20)
        self.e3 = nn.Linear(20, 10)

        self.d3 = nn.Linear(10, 20, bias=False)
        self.d2 = nn.Linear(20, 300, bias=False)
        self.d1 = nn.Linear(300, 784, bias=False)

        sizes = [300, 20, 10]
        decsizes = [20, 300, 784]

        self.g3 = MLPCombinator(10)
        self.g2 = MLPCombinator(20)
        self.g1 = MLPCombinator(300)
        self.g0 = MLPCombinator(784)

        self.acts = [self.relu, self.relu, self.softmax]
        self.layers = [self.e1, self.e2, self.e3]
        self.bnorms = [None] * len(self.layers)
        self.decoders = [self.d3, self.d2, self.d1]
        self.decbnorms = [None] * len(self.decoders)

        if 'encoder' in self.batchnorm_mode:
            for i in range(len(self.layers)):
                self.bnorms[i] = TwoStepBatchNorm(sizes[i])
                if self.cuda: 
                    self.bnorms[i].cuda()

        if 'decoder' in self.batchnorm_mode:
            for i in range(len(self.decoders)):
                self.decbnorms[i] = nn.BatchNorm1d(decsizes[i], affine=False)
                if self.cuda:
                    self.decbnorms[i].cuda()


    def bnorm(self, x):
        mean = x.mean(dim=1).expand(x.size())
        std = x.std(dim=1).expand(x.size())
        return (x - mean) / std


    def decode(self, layer, bnorm, g, recon, shortcut, z_norm=(0,1)):
        if not layer: # top most g has no predecessor layer
            u = recon
        else:
            u = layer(recon)
        u = bnorm(u) if (bnorm and 'decoder' in self.batchnorm_mode) else u
        z = g(u, shortcut)

        z_mu, z_std = z_norm
        z_bn = (z - z_mu) / z_std

        return z, z_bn


    def noise(self, x, noise_std):
        noise = Variable(torch.randn(x.size()) * noise_std)
        if self.cuda:
            noise = noise.cuda()
        return x + noise


    def encode(self, layer, bnorm, act, x, noise_std=0.):
        z = layer(x)

        if bnorm:
            z_mu, z_std, z, gamma, beta = bnorm(z)
            gamma = gamma.expand(z.size())
            beta = beta.expand(z.size())
        else:
            z_mu, z_std = 0, 1
            gamma, beta = 1, 0
    
        z = self.noise(z, noise_std) if noise_std > 0 else z

        h = z * gamma + beta
        z_act = act(h)

        return (z_mu,z_std), z, z_act


    def encode_all(self, e0, noise_std=0.):
        e0 = self.noise(e0, noise_std) if noise_std > 0 else e0

        e_act = e0
        pres = [e0]
        acts = []
        stat = [(0,1)]

        for (bnorm,layer,act) in zip(self.bnorms, self.layers, self.acts):
            e_stat, e_pre, e_act = self.encode(layer, bnorm, act, e_act, noise_std)
            stat += [e_stat]
            pres += [e_pre]
            acts += [e_act]

        return stat, pres, acts


    def forward(self, x):
        x = x.view(-1, 784)

        e_clean_stat, e_clean, e_clean_act = self.encode_all(x)
        e_noisy_stat, e_noisy, e_noisy_act = self.encode_all(x, self.noise_std)

        dbnorms = self.decbnorms
        l3_recon, l3_recon_bn = self.decode(None,          None, self.g3, e_noisy_act[-1], e_noisy[3], e_clean_stat[3])
        l2_recon, l2_recon_bn = self.decode(self.d3, dbnorms[0], self.g2, l3_recon, e_noisy[2], e_clean_stat[2])
        l1_recon, l1_recon_bn = self.decode(self.d2, dbnorms[1], self.g1, l2_recon, e_noisy[1], e_clean_stat[1])
        l0_recon, l0_recon_bn = self.decode(self.d1, dbnorms[2], self.g0, l1_recon, e_noisy[0], e_clean_stat[0])

        refs = e_clean
        recs = [l0_recon_bn, l1_recon_bn, l2_recon_bn, l3_recon_bn]
        sup_noisy = e_noisy_act[-1]
        sup_clean = e_clean_act[-1]

        return sup_noisy, sup_clean, (refs, recs)


    """
          y
          |   e3_rc
          o---------o
          |   e3_sc |   u3
         (f)   .-->(g)  g3
          o----'    |
    e3  [ooo]     [ooo] d3
          |         |   u2
         (f)   .-->(g)  g2
          o----'    |
    e2  [ooo]     [ooo] d2
          |         |   u1
         (f)   .-->(g)  g1
          o----'    |
    e1  [ooo]     [ooo] d1
          |         |   u0
          o------->(g)  g0
          |         |
    e0  [ooo]     [ooo]

    """

    def sample_by_example(self, x, k=1, use_bn=True):
        samples = []
        dbnorms = self.decbnorms

        for _ in range(k):
            e_noisy_stat, e_noisy, e_noisy_act = self.encode_all(x, self.noise_std)

            if use_bn:
                _, l3_recon = self.decode(None,          None, self.g3, e_noisy_act[-1], e_noisy[3], e_noisy_stat[3])
                _, l2_recon = self.decode(self.d3, dbnorms[0], self.g2, l3_recon, e_noisy[2], e_noisy_stat[2])
                _, l1_recon = self.decode(self.d2, dbnorms[1], self.g1, l2_recon, e_noisy[1], e_noisy_stat[1])
                _, l0_recon = self.decode(self.d1, dbnorms[2], self.g0, l1_recon, e_noisy[0], e_noisy_stat[0])
            else:
                l3_recon, _ = self.decode(None,          None, self.g3, e_noisy_act[-1], e_noisy[3], e_noisy_stat[3])
                l2_recon, _ = self.decode(self.d3, dbnorms[0], self.g2, l3_recon, e_noisy[2], e_noisy_stat[2])
                l1_recon, _ = self.decode(self.d2, dbnorms[1], self.g1, l2_recon, e_noisy[1], e_noisy_stat[1])
                l0_recon, _ = self.decode(self.d1, dbnorms[2], self.g0, l1_recon, e_noisy[0], e_noisy_stat[0])
            
            x = l0_recon

            samples.append(x)

        return samples


    def sample(self, z, k=1):
        """
        - we sample from \hat{P}(z) several times before using the reconstruction
        - we do this by repeatedly using the reconstruction as input

        References:
        - Improving Sampling from Generative Autoencoders with Markov Chains
        """
        e3_rc = e3_sc = z

        g3 = self.g3(e3_rc, e3_sc)
        u2 = self.d3(g3)
        #e3_sc = self.e3(self.relu(u2))

        # TODO

