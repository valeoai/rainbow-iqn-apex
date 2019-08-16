import random
import io
import math
import numpy as np
import torch

import rainbowiqn.constants as cst
from rainbowiqn.agent import Agent

class Actor(Agent):
    """This class just handle actor specific methods"""

    # Acts based on single state (no batch)
    def act(self, state_buffer):
        state = torch.from_numpy(np.stack(state_buffer).astype(np.float32) / 255).to(
            self.device, dtype=torch.float32
        )
        if self.rainbow_only:
            with torch.no_grad():
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
        else:
            with torch.no_grad():
                quantile_values, _ = self.online_net(state.unsqueeze(0), self.num_quantile_samples)
                return quantile_values.mean(0).argmax(0).item()

    def act_e_greedy(self, state_buffer, epsilon=0.001):
        """ Acts with an ε-greedy policy (used for evaluation only in test_multiple_seed.py)
        High ε can reduce eval score drastically"""
        return (
            random.randrange(self.action_space)
            if random.random() < epsilon
            else self.act(state_buffer)
        )

    def load_weight_from_redis(self):
        """Load weight from redis database"""
        load_bytesIO = io.BytesIO(self.redis_servor.get(cst.MODEL_WEIGHT_STR))
        self.online_net.load_state_dict(torch.load(load_bytesIO, map_location="cpu"))

    def compute_priorities(
        self, tab_state, tab_action, tab_reward, tab_nonterminal, priority_exponent
    ):
        """
        Compute priorities before sending experience to redis replay
        SOMETHING TO KEEP IN MIND, THE RETURNS CAN BE WRONG NEAR TERMINAL STATE AND
        IF REWARD AT BEGINNING OF EPISODE ARE NOT ZERO... (should we care?)
        In fact the states and next_states are also wrong near
        terminal state... it's kinda hard to
        take care of this properly, so we just
        initialize priorities badly around terminal state...
        """
        # REMINDER INDICE IN tab_state goes from -3 to len_buffer, the idea is that we got 3 more
        # states because we stack 4 states before sending it to network
        len_buffer = len(tab_action)
        assert (
            len(tab_action)
            == len(tab_reward)
            == len(tab_nonterminal)
            == len(tab_state) - self.history + 1
        )

        tab_nonterminal = np.float32(tab_nonterminal[self.n :])

        # Handling the case near a terminal state, indeed by construction the n-step states before
        # this terminal state are considered as terminal too (we want to know if in n-step the
        # episode will be already ended or not)
        current_term_indices = np.where(tab_nonterminal == 0)[0]
        for indice in current_term_indices:
            tab_nonterminal[indice + 1 : (indice + self.n + 1)] = 0

        actions = torch.tensor(
            tab_action[: len_buffer - self.n], dtype=torch.int64, device=self.device
        )
        tab_returns = [
            sum(self.discount ** n * tab_reward[n + indice] for n in range(self.n))
            for indice in range(0, len_buffer - self.n)
        ]
        returns = torch.tensor(tab_returns, dtype=torch.float32, device=self.device)
        nonterminals = torch.tensor(tab_nonterminal, dtype=torch.float32, device=self.device)

        tab_priorities = []
        for indice in range(math.ceil(len(actions) / self.batch_size)):
            current_begin = indice * self.batch_size
            current_end = min((indice + 1) * self.batch_size, len(actions))

            tab_current_state = []
            tab_current_next_state = []

            for current_sub_indice in range(current_begin, current_end):
                state = np.stack(tab_state[current_sub_indice : current_sub_indice + self.history])
                next_state = np.stack(
                    tab_state[
                        current_sub_indice + self.n : current_sub_indice + self.history + self.n
                    ]
                )

                tab_current_state.append(state)
                tab_current_next_state.append(next_state)

            states = (
                torch.from_numpy(np.stack(tab_current_state))
                .to(dtype=torch.float32, device=self.device)
                .div_(255)
            )
            next_states = (
                torch.from_numpy(np.stack(tab_current_next_state))
                .to(dtype=torch.float32, device=self.device)
                .div_(255)
            )

            loss = self.compute_loss_actor_or_learner(
                states,
                actions[current_begin:current_end],
                returns[current_begin:current_end],
                next_states,
                nonterminals[current_begin:current_end],
            )

            current_priorities = loss.detach().cpu().numpy()
            tab_priorities.append(current_priorities)
        priorities = np.concatenate(tab_priorities)

        return np.power(priorities, priority_exponent)
