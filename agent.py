import os
import random
import torch
from torch import optim
import numpy as np

from model import DQN

import io
import math

import CONSTANTS as CST
import compute_loss_iqn


class Agent():  # This class handle both actor and learner because most of their methods are shared
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
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
                print("We loaded model ", args.model)
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

        self.rainbow_only = args.rainbow_only

        if self.rainbow_only:  # Using standard Rainbow (i.e. C51 and no IQN)
            self.atoms = args.atoms
            self.Vmin = args.V_min
            self.Vmax = args.V_max
            self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
                device=args.device)  # Support (range) of z
            self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        else:  # Rainbow-IQN
            self.kappa = args.kappa
            self.num_tau_samples = args.num_tau_samples
            self.num_tau_prime_samples = args.num_tau_prime_samples
            self.num_quantile_samples = args.num_quantile_samples

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state_buffer):
        state = torch.from_numpy(np.stack(state_buffer).astype(np.float32) /
                                 255).to(self.device, dtype=torch.float32)
        if self.rainbow_only:
            with torch.no_grad():
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
        else:
            with torch.no_grad():
                quantile_values, _ = self.online_net(state.unsqueeze(0), self.num_quantile_samples)
                return quantile_values.mean(0).argmax(0).item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Compute loss for actor or learner, if it's for learner then we have to keep gradient flowing
    # but not if it's for actor which is just use to initialize priorities!
    def compute_loss_actor_or_learner(self, states, actions, returns, next_states, nonterminals):
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

                Tz = returns.unsqueeze(1) + nonterminals.unsqueeze(1) * (
                            self.discount ** self.n) * self.support.unsqueeze(
                    0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
                Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
                # Compute L2 projection of Tz onto fixed support z
                b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
                l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
                # Fix disappearing probability mass when l = b = u (b is int)
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1

                # Distribute probability of Tz
                m = states.new_zeros(batch_size, self.atoms)
                offset = torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size).unsqueeze(
                    1).expand(batch_size, self.atoms).to(actions)
                # m_l = m_l + p(s_t+n, a*)(u - b)
                m.view(-1).index_add_(0, (l + offset).view(-1),
                                      (pns_a * (u.float() - b)).view(-1))
                # m_u = m_u + p(s_t+n, a*)(b - l)
                m.view(-1).index_add_(0, (u + offset).view(-1),
                                      (pns_a * (b - l.float())).view(-1))

                ##################################################
                # Compute target quantile values, so no gradient #
                # there in both case (actor or learner)          #
                ##################################################

            loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimise DKL(m||p(s_t, a_t)))

        else:  # IQN loss
            loss = compute_loss_iqn.compute_loss_actor_or_learner_iqn(
                self, states, actions, returns, next_states, nonterminals)
        return loss

    ########################
    #   ONLY FOR LEARNER   #
    ########################

    def learn(self, mem_redis, mp_queue):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights\
            = mem_redis.get_sample_from_mp_queue(mp_queue)

        loss = self.compute_loss_actor_or_learner(states, actions,
                                                  returns, next_states, nonterminals)

        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Importance weight losses
        self.optimiser.step()

        return idxs, loss

        # priorities = loss.detach().cpu().numpy()
        #
        # return idxs, priorities

    # Acts with an ε-greedy policy (used for evaluation only in test_multiple_seed.py)
    def act_e_greedy(self, state_buffer, epsilon=0.001):  # High ε can reduce eval score drastically
        return random.randrange(self.action_space) if random.random() < epsilon else self.act(state_buffer)

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    # Store weight into redis servor
    def save_to_redis(self, T_learner):
        # test_state_dict_before = copy.deepcopy(self.online_net.state_dict())

        # ONLY PART WHICH IS NOT TO REMOVE
        save_bytesIO = io.BytesIO()
        torch.save(self.online_net.state_dict(), save_bytesIO)
        pipe = self.redis_servor.pipeline()
        pipe.set(CST.MODEL_WEIGHT_STR, save_bytesIO.getvalue())
        pipe.set(CST.STEP_LEARNER_STR, T_learner)
        pipe.execute()
        # ONLY PART WHICH IS NOT TO REMOVE

        # load_bytesIO = io.BytesIO(self.redis_servor.get(CST.MODEL_WEIGHT_STR))
        # self.online_net.load_state_dict(torch.load(load_bytesIO, map_location='cpu'))
        # test_state_dict_after = self.online_net.state_dict()
        #
        # print("we test if before and after are the same! TODO TO REMOVE")
        # itemsbefore = list(test_state_dict_before.items())
        # itemsafter = list(test_state_dict_after.items())
        # assert len(itemsbefore) == len(itemsafter)
        # for ind_item in range(len(itemsbefore)):
        #     assert itemsbefore[ind_item][0] == itemsafter[ind_item][0]
        #     assert int(torch.min(itemsbefore[ind_item][1] == itemsafter[ind_item][1])) == 1

    ########################
    ### ONLY FOR LEARNER ###
    ########################

    ######################
    ### ONLY FOR ACTOR ###
    ######################

    # Load weight from redis database
    def load_weight_from_redis(self):
        load_bytesIO = io.BytesIO(self.redis_servor.get(CST.MODEL_WEIGHT_STR))
        self.online_net.load_state_dict(torch.load(load_bytesIO, map_location='cpu'))

    # Compute priorities before sending experience to redis replay
    # SOMETHING TO KEEP IN MIND, THE RETURNS CAN BE WRONG NEAR TERMINAL STATE AND IF REWARD AT BEGINNING OF EPISODE ARE NOT ZERO... (should we care?)
    # In fact the states and next_states too are wrong near terminal state... it's kinda hard to take care of this properly, so we just initiliaze priorities pretty badly around terminal state...
    def compute_priorities(self, tab_state, tab_action, tab_reward, tab_nonterminal, priority_exponent):
        # REMINDER INDICE IN tab_state goes from -3 to len_buffer, the idea is that we got 3 more states because we stack 4 states before sending it to network
        len_buffer = len(tab_action)
        assert len(tab_action) == len(tab_reward) == len(tab_nonterminal) == len(tab_state) - self.history + 1

        tab_nonterminal = np.float32(tab_nonterminal[self.n:])

        # Handling the case near a terminal state, indeed by construction the n-step states before this terminal state
        # are considered as terminal too (we want to know if in n-step the episode will be already ended or not)
        current_term_indices = np.where(tab_nonterminal == 0)[0]
        for indice in current_term_indices:
            tab_nonterminal[indice+1:(indice + self.n + 1)] = 0

        actions = torch.tensor(tab_action[:len_buffer - self.n], dtype=torch.int64, device=self.device)
        tab_returns = [sum(self.discount ** n * tab_reward[n + indice] for n in range(self.n)) for indice in range(0, len_buffer - self.n)]
        returns = torch.tensor(tab_returns, dtype=torch.float32, device=self.device)
        nonterminals = torch.tensor(tab_nonterminal, dtype=torch.float32, device=self.device)

        tab_priorities = []
        for indice in range(math.ceil(len(actions)/self.batch_size)):
            current_begin = indice * self.batch_size
            current_end = min((indice+1) * self.batch_size, len(actions))

            tab_current_state = []
            tab_current_next_state = []

            for current_sub_indice in range(current_begin, current_end):
                state = np.stack(tab_state[current_sub_indice:current_sub_indice+self.history])
                next_state = np.stack(tab_state[current_sub_indice+self.n:current_sub_indice + self.history + self.n])

                tab_current_state.append(state)
                tab_current_next_state.append(next_state)

            states = torch.from_numpy(np.stack(tab_current_state)).to(dtype=torch.float32, device=self.device).div_(255)
            next_states = torch.from_numpy(np.stack(tab_current_next_state)).to(dtype=torch.float32, device=self.device).div_(255)

            loss = self.compute_loss_actor_or_learner(states,
                                                      actions[current_begin:current_end],
                                                      returns[current_begin:current_end],
                                                      next_states,
                                                      nonterminals[current_begin:current_end])

            current_priorities = loss.detach().cpu().numpy()
            tab_priorities.append(current_priorities)
        priorities = np.concatenate(tab_priorities)

        return np.power(priorities, priority_exponent)

    ######################
    ### ONLY FOR ACTOR ###
    ######################

