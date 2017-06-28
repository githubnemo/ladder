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

        self.l1 = nn.Linear(3*input_shape, input_shape)
        self.l2 = nn.Linear(4*input_shape, input_shape)

        self.sig = nn.Sigmoid()

    def forward(self, u, z):
        x = torch.cat((u, z, u*z), dim=1)
        s = self.sig(self.l1(x))
        return self.l2(torch.cat((x,s), dim=1))


class Encoder(nn.Module):
    def __init__(self, layers, acts, cuda, noise_std=0.):
        super().__init__()
        self.cuda = cuda
        self.acts = acts
        self.layers = layers
        self.noise_std = noise_std

    def noise(self, x):
        noise = Variable(torch.randn(x.size()) * self.noise_std)
        if self.cuda:
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
    def __init__(self, parallel_encoder=False, noise_std=0.3, cuda=False):
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

        self.e_noisy = Encoder(layers=layers, acts=acts, cuda=cuda, noise_std=noise_std)
        self.e_clean = Encoder(layers=layers, acts=acts, cuda=cuda, noise_std=0.0)

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


    """

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
    def sample_k(self, z, k=1):
        e3_rc = z
        e3_sc = self.e_noisy.noise(z)

        for _ in range(k):
            g3 = self.g3(e3_rc, e3_sc)
            u2 = self.d3(g3)
            e3_sc = self.e_noisy.noise(self.e3(u2))

        e2_rc = u2
        e2_sc = self.e_noisy.noise(u2)

        for _ in range(k):
            g2 = self.g2(e2_rc, e2_sc)
            u1 = self.d2(g2)
            e2_sc = self.e_noisy.noise(self.e2(u1))

        e1_rc = u1
        e1_sc = self.e_noisy.noise(u1)

        for _ in range(k):
            g1 = self.g1(e1_rc, e1_sc)
            u0 = self.d1(g1)
            e1_sc = self.e_noisy.noise(self.e1(u0))

        e0_rc = u0
        e0_sc = self.e_noisy.noise(u0)

        g0 = self.g0(e0_rc, e0_sc)

        return g0

    def sample_s(self, z, j=1, k=None):
        x = self.d1(self.d2(self.d3(z)))

        for _ in range(j):
            (e0_noisy,e1_noisy,e2_noisy,_), (e1_noisy_act,e2_noisy_act,_) = self.e_clean(x)

            l3_recon = z
            u2 = self.d3(l3_recon)
            l2_recon = self.g2(u2, e2_noisy)
            u1 = self.d2(l2_recon)
            l1_recon = self.g1(u1, e1_noisy)
            u0 = self.d1(l1_recon)
            l0_recon = self.g0(u0, e0_noisy)

            x = l0_recon

            mean = x.mean(dim=1).expand(x.size())
            std = x.std(dim=1).expand(x.size())
            x = (x)/std
        return x

    def sample_j(self, z, j=1, k=1):
        for _ in range(j):
            x = self.sample_k(z, k)
            (e0_clean,e1_clean,e2_clean,e3_clean), (e1_clean_act,e2_clean_act,e3_clean_act) = self.e_clean(x)
            z = e3_clean_act
        return x

    def sample_ble(self, z, k=1, j=1):
        x = self.d1(self.d2(self.d3(z)))

        (e0_noisy,e1_noisy,e2_noisy,e3_noisy), (e1_noisy_act,e2_noisy_act,e3_noisy_act) = self.e_noisy(x)

        for _ in range(j):
            u2 = self.d3(z)

            for _ in range(k):
                l2_recon = self.g2(u2, e2_noisy)
                u1 = self.d2(l2_recon)
                e2_noisy, _ = self.e_noisy.encode(self.e2, self.relu, u1)

            for _ in range(k):
                l2_recon = self.g2(u2, e2_noisy)
                u1 = self.d2(l2_recon)
                l1_recon = self.g1(u1, e1_noisy)
                u0 = self.d1(l1_recon)
                e1_noisy, _ = self.e_noisy.encode(self.e1, self.relu, u0)
                e2_noisy, _ = self.e_noisy.encode(self.e2, self.relu, e1_noisy)

            for _ in range(k):
                l2_recon = self.g2(u2, e2_noisy)
                u1 = self.d2(l2_recon)
                l1_recon = self.g1(u1, e1_noisy)
                u0 = self.d1(l1_recon)
                l0_recon = self.g0(u0, e0_noisy)
                e0_noisy = l0_recon
                e1_noisy, _ = self.e_noisy.encode(self.e1, self.relu, e0_noisy)
                e2_noisy, _ = self.e_noisy.encode(self.e2, self.relu, e1_noisy)

            x = l0_recon
            continue


            l3_recon = z
            u2 = self.d3(l3_recon)
            l2_recon = self.g2(u2, e2_noisy)
            u1 = self.d2(l2_recon)
            l1_recon = self.g1(u1, e1_noisy)
            u0 = self.d1(l1_recon)
            l0_recon = self.g0(u0, e0_noisy)

            x = l0_recon

        return x

    def sample(self, z, k=1):
        z_noisy_1 = self.e_noisy.noise(z)
        z_noisy_2 = self.e_noisy.noise(z)

        for _ in range(k):
            l3_recon = self.g3(z_noisy_1, z_noisy_2)
            z_noisy_2 = l3_recon


        u2_1 = self.d3(l3_recon)
        u2_2 = self.e_noisy.noise(u2_1)

        for _ in range(k):
            l2_recon = self.g2(u2_1, u2_2)
            u2_2 = self.e_noisy.noise(l2_recon)

        u1_1 = self.d2(l2_recon)
        u1_2 = self.e_noisy.noise(u1_1)

        for _ in range(k):
            l1_recon = self.g1(u1_1, u1_2)
            u1_2 = self.e_noisy.noise(l1_recon)

        u0_1 = self.d1(l1_recon)
        u0_2 = self.e_noisy.noise(u0_1)

        for _ in range(k):
            l0_recon = self.g0(u0_1, u0_2)
            u0_2 = self.e_noisy.noise(l0_recon)

        return l0_recon

