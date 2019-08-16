import math

import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init, disable_cuda=False):
        super().__init__()
        self.disable_cuda = disable_cuda
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        if self.disable_cuda:
            x = torch.randn(size)
        else:
            x = torch.cuda.FloatTensor(size).normal_()
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.rainbow_only = args.rainbow_only

        self.action_space = action_space
        self.device = args.device
        self.disable_cuda = args.disable_cuda

        self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        if self.rainbow_only:  # Model is different if using Rainbow only or using Rainbow IQN
            self.atoms = args.atoms
            # We are looking for substring "fcnoisy" in name to detect which layer are noisy layer!
            self.fcnoisy_h_v = NoisyLinear(
                3136, args.hidden_size, std_init=args.noisy_std, disable_cuda=args.disable_cuda
            )
            self.fcnoisy_h_a = NoisyLinear(
                3136, args.hidden_size, std_init=args.noisy_std, disable_cuda=args.disable_cuda
            )
            self.fcnoisy_z_v = NoisyLinear(
                args.hidden_size,
                self.atoms,
                std_init=args.noisy_std,
                disable_cuda=args.disable_cuda,
            )
            self.fcnoisy_z_a = NoisyLinear(
                args.hidden_size,
                action_space * self.atoms,
                std_init=args.noisy_std,
                disable_cuda=args.disable_cuda,
            )

        else:  # Rainbow-IQN model
            self.quantile_embedding_dim = args.quantile_embedding_dim
            self.iqn_fc = nn.Linear(self.quantile_embedding_dim, 3136)

            # We are looking for substring "fcnoisy" in name to detect which layer are noisy layer!
            self.fcnoisy_h_v = NoisyLinear(
                3136, args.hidden_size, std_init=args.noisy_std, disable_cuda=args.disable_cuda
            )
            self.fcnoisy_h_a = NoisyLinear(
                3136, args.hidden_size, std_init=args.noisy_std, disable_cuda=args.disable_cuda
            )
            self.fcnoisy_z_v = NoisyLinear(
                args.hidden_size, 1, std_init=args.noisy_std, disable_cuda=args.disable_cuda
            )
            self.fcnoisy_z_a = NoisyLinear(
                args.hidden_size,
                action_space,
                std_init=args.noisy_std,
                disable_cuda=args.disable_cuda,
            )

    def forward(self, x, num_quantiles=None, log=False):
        batch_size = x.shape[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)

        if self.rainbow_only:
            v = self.fcnoisy_z_v(F.relu(self.fcnoisy_h_v(x)))  # Value stream
            a = self.fcnoisy_z_a(F.relu(self.fcnoisy_h_a(x)))  # Advantage stream
            v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
            q = v + a - a.mean(1, keepdim=True)  # Combine streams
            if log:  # Use log softmax for numerical stability
                q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
            else:
                q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
            return q
        else:
            if self.disable_cuda:
                quantiles = torch.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1)
            else:
                quantiles = torch.cuda.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1)

            quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])

            quantile_net = torch.cos(
                torch.arange(
                    1, self.quantile_embedding_dim + 1, 1, device=self.device, dtype=torch.float32
                )
                * math.pi
                * quantile_net
            )

            quantile_net = self.iqn_fc(quantile_net)
            quantile_net = F.relu(quantile_net)

            x = x.repeat(num_quantiles, 1)

            x = x * quantile_net

            v = self.fcnoisy_z_v(F.relu(self.fcnoisy_h_v(x)))  # Value stream
            a = self.fcnoisy_z_a(F.relu(self.fcnoisy_h_a(x)))  # Advantage stream

            q = v + a - a.mean(1, keepdim=True)  # Combine streams
            return q, quantiles

    def reset_noise(self):
        for name, module in self.named_children():
            if "fcnoisy" in name:
                module.reset_noise()
