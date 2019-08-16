#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:20:12 2018

@author: toromanoff
"""

import random
import torch

from rainbowiqn.env import Env

from rainbowiqn.agent import Agent


from rainbowiqn.args import return_args

import time
import numpy as np


# Test the input snapshot (args --model)
def test_multiple_seed(args, debug=False):
    T_rewards = []
    T_rewards_5minutes = []
    T_true_RL_reward = []  # This is the actual reward the agent see, can be really different
    # from the game score because we clip the reward between -1 and 1 by default!
    epsilon = 0  # We got sticky actions now
    # Test performance over several episodes varying the random seed
    done = True

    for episode_number in range(args.evaluation_episodes):
        args.seed += 1
        random.seed(args.seed)
        torch.manual_seed(random.randint(1, 10000))
        env = Env(args)
        tester = Agent(args, env.action_space(), None)
        step = 0
        while True:
            step += 1
            if done:
                state_buffer = env.reset()
                reward_sum, done = 0, False
                true_RL_reward_sum = 0

            # Choose an action ε-greedily
            action = tester.act_e_greedy(state_buffer, epsilon=epsilon)
            state_buffer, reward, done = env.step(action)  # Step

            # current_image = state_buffer[-1]
            # cv2.imshow("current_image", current_image)
            # cv2.waitKey(1)
            if step % args.replay_frequency == 0:
                tester.reset_noise()  # Draw a new set of noisy weights

            # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
            if step == int(5 * 60 * 60 / args.action_repeat):
                T_rewards_5minutes.append(reward_sum)

            reward_sum += reward
            true_RL_reward_sum += np.clip(reward, -1, 1)
            if debug and step % 5000 == 0:
                time.sleep(0.1)
                print("step = ", step)
                print("reward_sum = ", reward_sum)

            if args.render:
                env.render()

            if done:
                if debug:
                    print(
                        "Episode terminated after "
                        + str(step)
                        + " steps. Reward episode "
                        + str(episode_number)
                        + " = ",
                        reward_sum,
                    )
                T_rewards.append(reward_sum)
                T_true_RL_reward.append(true_RL_reward_sum)
                # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
                if step <= int(5 * 60 * 60 / args.action_repeat):
                    T_rewards_5minutes.append(reward_sum)
                break
    env.close()

    avg_reward = sum(T_rewards) / len(T_rewards)

    if debug:
        print(f"avg_reward on {args.evaluation_episodes} is {avg_reward}")

    return avg_reward


if __name__ == "__main__":
    args = return_args()
    args.id_actor = 0
    test_multiple_seed(args, True)
