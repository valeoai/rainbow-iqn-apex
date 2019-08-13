import os
import random
import torch
from torch import nn, optim
import numpy as np

from model import DQN

import time
import io
import math

import CONSTANTS as CST

class Agent(): # This class handle both actor and learner because most of their methods are shared
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
        
        self.kappa = args.kappa
        self.num_tau_samples = args.num_tau_samples
        self.num_tau_prime_samples = args.num_tau_prime_samples
        self.num_quantile_samples = args.num_quantile_samples

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state_buffer):
        state = torch.from_numpy(np.stack(state_buffer).astype(np.float32) / 255).to(self.device, dtype=torch.float32)
        with torch.no_grad():  
            quantile_values, _ = self.online_net(state.unsqueeze(0), self.num_quantile_samples)
            return quantile_values.mean(0).argmax(0).item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    #  Compute loss for actor or learner, if it's for learner then we have to keep gradient flowing but not if it's for actor which is just use to initialize priorities!
    def compute_loss_actor_or_learner(self, states, actions, returns, next_states, nonterminals):

        batch_size = len(states)

        with torch.no_grad():
            
            ##################################################
            # Compute target quantile values, so no gradient #
            # there in both case (actor or learner)          #
            ##################################################
            
            # Shape of returns will be (num_tau_prime_samples x batch_size) x 1
            returns = returns[:,None].repeat([self.num_tau_prime_samples, 1])
            
            # Shape of gamma_with_terminal will be (num_tau_prime_samples x batch_size) x 1
            gamma_with_terminal = (self.discount ** self.n) * nonterminals[:,None]
            gamma_with_terminal = gamma_with_terminal.repeat([self.num_tau_prime_samples, 1])

#            print("gamma_with_terminal.shape = ", gamma_with_terminal.shape)
            
            # Compute target quantiles values Q(s_t+n, num_quantile_samples; θonline)
            # We used online_net there cause we use double DQN
            self.online_net.reset_noise()
            target_quantile_values_action, _ = self.online_net(next_states, self.num_quantile_samples)
            target_quantile_values_action = target_quantile_values_action.reshape([
                    self.num_quantile_samples, batch_size, self.action_space])
    
            replay_net_target_q_values = torch.mean(target_quantile_values_action, dim = 0)
#            print("replay_net_target_q_values.shape = ", replay_net_target_q_values.shape)
            
            # Perform argmax action selection using online network: argmax_a[Q(s_t+n; θonline)]
            replay_next_qt_argmax = torch.argmax(replay_net_target_q_values, dim = 1)
#            print("replay_next_qt_argmax.shape = ", replay_next_qt_argmax.shape)
            
            # Shape of replay_next_qt_argmax will be (num_tau_prime_samples x batch_size) x 1
            replay_next_qt_argmax = replay_next_qt_argmax[:,None].repeat([self.num_tau_prime_samples, 1])
#            print("replay_next_qt_argmax.shape = ", replay_next_qt_argmax.shape)
            
            # Compute target quantiles values Q(s_t+n, num_tau_prime_samples; θtarget)
            # Shape of replay_net_target_quantile_values will be (num_tau_prime_samples x batch_size) x action_space
            self.target_net.reset_noise()
            replay_net_target_quantile_values, _ = self.target_net(next_states, self.num_tau_prime_samples)
#            print("replay_net_target_quantile_values.shape = ", replay_net_target_quantile_values.shape)
            
            # Double-Q values Q(s_t+n, argmax_a[Q(s_t+n; θonline)]; θtarget)
            # Shape of target_quantile_values will be (num_tau_prime_samples x batch_size) x 1
            target_quantile_values = torch.gather(replay_net_target_quantile_values, 1, replay_next_qt_argmax)
#            print("target_quantile_values.shape = ", target_quantile_values.shape)

            # Shape of full_target_quantile_values will be (num_tau_prime_samples x batch_size) x 1
            full_target_quantile_values = returns + gamma_with_terminal * target_quantile_values
#            print("full_target_quantile_values.shape = ", full_target_quantile_values.shape)
            
            # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
            # the manner in which the target_quantile_values are tiled.         
            full_target_quantile_values = full_target_quantile_values.reshape([self.num_tau_prime_samples,batch_size, 1])
#            print("full_target_quantile_values.shape = ", full_target_quantile_values.shape)
            
            # Transpose dimensions so that the dimensionality is batch_size x
            # self.num_tau_prime_samples x 1 to prepare for computation of
            # Bellman errors.
            # Final shape of full_target_quantile_values:
            # batch_size x num_tau_prime_samples x 1.
            full_target_quantile_values = full_target_quantile_values.permute([1, 0, 2])
#            print("full_target_quantile_values.shape = ", full_target_quantile_values.shape)
            
            ##################################################
            # Compute target quantile values, so no gradient #
            # there in both case (actor or learner)          #
            ##################################################
            
        # Compute current quantiles values Q(s_t, num_tau_samples; θonline)
        # Shape of replay_net_quantile_values will be (num_tau_samples x batch_size) x action_space
        self.online_net.reset_noise()
        replay_net_quantile_values, replay_quantiles = self.online_net(states, self.num_tau_samples)
#        print("replay_net_quantile_values.shape = ", replay_net_quantile_values.shape)

        # Shape of actions will be (num_tau_samples x batch_size) x 1
        actions = actions[:,None].repeat([self.num_tau_samples, 1])
#        print("actions.shape = ", actions.shape)
        
        # Compute current quantiles values Q(s_t, actions; θnline)
        # Shape of chosen_action_quantile_values will be (num_tau_samples x batch_size) x 1
        chosen_action_quantile_values = torch.gather(replay_net_quantile_values, 1, actions)
#        print("chosen_action_quantile_values.shape = ", chosen_action_quantile_values.shape)

        # Reshape to self.num_tau_samples x batch_size x 1 since this is
        # the manner in which the target_quantile_values are tiled.         
        chosen_action_quantile_values = chosen_action_quantile_values.reshape([self.num_tau_samples,batch_size, 1])
#        print("chosen_action_quantile_values.shape = ", chosen_action_quantile_values.shape)

        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_samples x 1 to prepare for computation of
        # Bellman errors.
        # Final shape of chosen_action_quantile_values:
        # batch_size x num_tau_samples x 1.            
        chosen_action_quantile_values = chosen_action_quantile_values.permute([1, 0, 2])
#        print("chosen_action_quantile_values.shape = ", chosen_action_quantile_values.shape)

        # Shape of bellman_erors and huber_loss:
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        bellman_errors = full_target_quantile_values[:, :, None, :] - chosen_action_quantile_values[:, None, :, :]
#        print("bellman_errors.shape = ", bellman_errors.shape)

        # The huber loss (see Section 2.3 of the paper) is defined via two cases:
        # case_one: |bellman_errors| <= kappa
        # case_two: |bellman_errors| > kappa
        huber_loss_case_one = (torch.abs(bellman_errors) <= self.kappa).float() * 0.5 * bellman_errors ** 2
        huber_loss_case_two = (torch.abs(bellman_errors) > self.kappa).float() * self.kappa * (torch.abs(bellman_errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two
#        print("huber_loss.shape = ", huber_loss.shape)
        
        # Reshape replay_quantiles to batch_size x num_tau_samples x 1
#        replay_quantiles = torch.from_numpy(replay_quantiles).to(self.device, dtype=torch.float32)
        replay_quantiles = torch.reshape(replay_quantiles, [self.num_tau_samples, batch_size, 1])
#        print("replay_quantiles.shape = ", replay_quantiles.shape)
        replay_quantiles = replay_quantiles.permute([1, 0, 2])
#        print("replay_quantiles.shape = ", replay_quantiles.shape)

        # Tile by num_tau_prime_samples along a new dimension. Shape is now
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        # These quantiles will be used for computation of the quantile huber loss
        # below (see section 2.3 of the paper).
        replay_quantiles = replay_quantiles[:, None, :, :].repeat([1, self.num_tau_prime_samples, 1, 1])
#        print("replay_quantiles.shape = ", replay_quantiles.shape)
        
        # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
        quantile_huber_loss = (torch.abs(replay_quantiles - ((bellman_errors < 0).float()).detach()) * huber_loss) / self.kappa
#        print("quantile_huber_loss.shape = ", quantile_huber_loss.shape)
        
        # Sum over current quantile value (num_tau_samples) dimension,
        # average over target quantile value (num_tau_prime_samples) dimension.
        # Shape: batch_size x num_tau_prime_samples x 1.
        loss = torch.sum(quantile_huber_loss, dim=2)
#        print("loss.shape = ", loss.shape)
        # Shape: batch_size x 1.
        loss = torch.mean(loss, dim=1)
#        print("loss.shape = ", loss.shape)

        # Shape: batch_size.
        loss = loss[:,0]
#        print("loss.shape = ", loss.shape)
        return loss

    ########################
    ### ONLY FOR LEARNER ###
    ########################

    def learn(self, mem_redis, mp_queue):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem_redis.get_sample_from_mp_queue(mp_queue)

        loss = self.compute_loss_actor_or_learner(states, actions, returns, next_states, nonterminals)

        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Importance weight losses
        self.optimiser.step()

        return idxs, loss

        # priorities = loss.detach().cpu().numpy()
        #
        # return idxs, priorities

    # Acts with an ε-greedy policy (used for evaluation only in test_multiple_seed.py)
    def act_e_greedy(self, state_buffer, epsilon=0.001):  # High ε can reduce evaluation scores drastically
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
        # REMINDER INDICE IN tab_state goes from -3 to len_buffer, the idea is that we got 3 more states cause we stack 4 states before sending it to network
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

