import io
import math
import os
import random

import numpy as np
import torch
from torch import optim

import rainbowiqn.compute_loss_iqn as compute_loss_iqn
import rainbowiqn.constants as cst
from rainbowiqn.model import DQN


class Agent:
    """This class handle both actor and learner because most of their methods are shared"""

    def __init__(self, args, action_space, redis_servor):
        self.action_space = action_space

        self.n = args.multi_step
        self.history = args.history_length
        self.discount = args.discount
        self.redis_servor = redis_servor
        self.device = args.device

        self.batch_size = args.batch_size
        self.length_actor_buffer = args.length_actor_buffer

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:
            if os.path.isfile(args.model):
                print("We loaded model ", args.model)
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                checkpoint = torch.load(args.model, map_location="cpu")
                self.online_net.load_state_dict(checkpoint["model_state_dict"])
            else:
                print("We didn't fint the model you gave as input!")
                raise Exception
        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

        if args.model:
            # We already loaded the checkpoint there, no need to load it again
            self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

        self.rainbow_only = args.rainbow_only

        if self.rainbow_only:  # Using standard Rainbow (i.e. C51 and no IQN)
            self.atoms = args.atoms
            self.Vmin = args.V_min
            self.Vmax = args.V_max
            self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
                device=args.device
            )  # Support (range) of z
            self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        else:  # Rainbow-IQN
            self.kappa = args.kappa
            self.num_tau_samples = args.num_tau_samples
            self.num_tau_prime_samples = args.num_tau_prime_samples
            self.num_quantile_samples = args.num_quantile_samples

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def compute_loss_actor_or_learner(self, states, actions, returns, next_states, nonterminals):
        """Compute loss for actor or learner, if it's for learner then
        we have to keep gradient flowing but not if it's for actor which is
        just use to initialize priorities!
        """
        if self.rainbow_only:  # Rainbow only loss (C51 in stead of IQN)
            batch_size = len(states)

            # Calculate current state probabilities (online network noise already sampled)
            log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
            log_ps_a = log_ps[range(batch_size), actions]  # log p(s_t, a_t; θonline)

            with torch.no_grad():

                ##################################################
                # Compute target quantile values, so no gradient #
                # there in both case (actor or learner)          #
                ##################################################

                # Calculate nth next state probabilities
                # Probabilities p(s_t+n, ·; θonline)
                pns = self.online_net(next_states)
                # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
                dns = self.support.expand_as(pns) * pns
                # Perform argmax action selection using online network:
                # argmax_a[(z, p(s_t+n, a; θonline))]
                argmax_indices_ns = dns.sum(2).argmax(1)
                self.target_net.reset_noise()  # Sample new target net noise
                pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
                # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                pns_a = pns[range(batch_size), argmax_indices_ns]

                # Compute Tz (Bellman operator T applied to z)

                # Tz = R^n + (γ^n)z (accounting for terminal states)
                Tz = returns.unsqueeze(1) + nonterminals.unsqueeze(1) * (
                    self.discount ** self.n
                ) * self.support.unsqueeze(0)
                Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
                # Compute L2 projection of Tz onto fixed support z
                b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
                l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
                # Fix disappearing probability mass when l = b = u (b is int)
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1

                # Distribute probability of Tz
                m = states.new_zeros(batch_size, self.atoms)
                offset = (
                    torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size)
                    .unsqueeze(1)
                    .expand(batch_size, self.atoms)
                    .to(actions)
                )
                # m_l = m_l + p(s_t+n, a*)(u - b)
                m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
                # m_u = m_u + p(s_t+n, a*)(b - l)
                m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

                ##################################################
                # Compute target quantile values, so no gradient #
                # there in both case (actor or learner)          #
                ##################################################

            # Cross-entropy loss (minimise DKL(m||p(s_t, a_t)))
            loss = -(torch.sum(m * log_ps_a, 1))

        else:  # IQN loss
            loss = compute_loss_iqn.compute_loss_actor_or_learner_iqn(
                self, states, actions, returns, next_states, nonterminals
            )
        return loss

    # Save model parameters on results folder (or on --path-to-results given)
    def save(self, path, T_actors, T_learner, name):
        # torch.save(self.online_net.state_dict(), os.path.join(path, name))
        torch.save(
            {
                "T_actors": T_actors,
                "T_learner": T_learner,
                "model_state_dict": self.online_net.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
            },
            os.path.join(path, name),
        )

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
