import time
from datetime import datetime

from rainbowiqn.env import Env

from rainbowiqn.redis_memory import ReplayRedisMemory

from rainbowiqn.agent import Agent

from torch.multiprocessing import Process

import redis
import logging
from rainbowiqn.args import return_args

import numpy as np

import rainbowiqn.constants as cst

from collections import deque

from rainbowiqn.utils import _plot_line, dump_in_csv
import random

import os


# Simple ISO 8601 timestamped logger
def log(s):
    print("[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s)


# Send actor buffer experience to main memory
def send_actor_buffer(
    actor_buffer, actor_index_in_replay_memory, id_actor, mem_actor, priorities, T_actor
):
    """actor_index_in_replay_memory is the index
    in the memory where to start appending next buffer
    """
    mem_actor.transitions.append_actor_buffer(
        actor_buffer, actor_index_in_replay_memory, id_actor, priorities, T_actor
    )


# Create an actor instance
def launch_actor(id_actor, args, redis_servor):

    print("id actor = ", id_actor)
    env_actor = Env(args)

    start_time_actor = time.time()

    if args.continue_experiment:
        print(
            "We are restarting a stopped experience with a model trained for "
            + str(args.step_actors_already_done)
            + "steps"
        )
        initial_T_actor = int(
            (args.step_actors_already_done - args.memory_capacity) / args.nb_actor
        )
        print("initial T actor equal ", initial_T_actor)
        step_to_start_sleep = int(args.step_actors_already_done / args.nb_actor)
    else:
        initial_T_actor = 0
        step_to_start_sleep = int(args.learn_start / args.nb_actor)
    T_actor = initial_T_actor

    index_actor_in_memory = 0
    timestep = 0
    actor_buffer = []
    mem_actor = ReplayRedisMemory(args, redis_servor)

    actor = Agent(args, env_actor.action_space(), redis_servor)

    done_actor = True

    tab_state = []
    tab_action = []
    tab_reward = []
    tab_nonterminal = []

    if id_actor == 0:
        # Variables for plot and dump in csv only
        Tab_T_actors, Tab_T_learner, Tab_length_episode, Tab_longest_episode = ([], [], [], [])
        tab_rewards_plot, best_avg_reward = [], -1e10

        # We initialize all buffer with 0 because sometimes there are not totally
        # filled for the first evaluation step and this leads to a bug in the plot...
        total_reward_buffer_SABER = deque(
            [0] * args.evaluation_episodes, maxlen=args.evaluation_episodes
        )
        total_reward_buffer_30min = deque(
            [0] * args.evaluation_episodes, maxlen=args.evaluation_episodes
        )
        total_reward_buffer_5min = deque(
            [0] * args.evaluation_episodes, maxlen=args.evaluation_episodes
        )
        episode_length_buffer = deque(
            [0] * args.evaluation_episodes, maxlen=args.evaluation_episodes
        )
        current_total_reward_SABER = 0
        current_total_reward_30min = 0
        current_total_reward_5min = 0

    while T_actor < (args.T_max / args.nb_actor):
        if done_actor:
            if id_actor == 0 and T_actor > initial_T_actor:
                total_reward_buffer_SABER.append(current_total_reward_SABER)

                if (
                    timestep < (5 * 60 * 60) / args.action_repeat
                ):  # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
                    current_total_reward_5min = current_total_reward_SABER
                total_reward_buffer_5min.append(current_total_reward_5min)

                if (
                    timestep < (30 * 60 * 60) / args.action_repeat
                ):  # 30 minutes * 60 secondes * 60 HZ Atari game / action repeat
                    current_total_reward_30min = current_total_reward_SABER
                total_reward_buffer_30min.append(current_total_reward_30min)

                episode_length_buffer.append(timestep)
                current_total_reward_SABER = 0
                current_total_reward_30min = 0
                current_total_reward_5min = 0
            timestep = 0
            state_buffer_actor = env_actor.reset()
            done_actor = False

        if T_actor % args.replay_frequency == 0:
            actor.reset_noise()  # Draw a new set of noisy weights

        if T_actor < args.learn_start / args.nb_actor:  # Do random actions before learning start
            action = random.randint(0, env_actor.action_space() - 1)
        else:
            action = actor.act(
                state_buffer_actor
            )  # Choose an action greedily (with noisy weights)

        next_state_buffer_actor, reward, done_actor = env_actor.step(action)  # Step
        if args.render and id_actor == 0:
            env_actor.render()

        if id_actor == 0:
            # THIS should be before clipping, we want to know the true score of the game there!
            current_total_reward_SABER += reward
            # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
            if timestep == (5 * 60 * 60) / args.action_repeat:
                current_total_reward_5min = current_total_reward_SABER
            # 30 minutes * 60 secondes * 60 HZ Atari game / action repeat
            if timestep == (30 * 60 * 60) / args.action_repeat:
                current_total_reward_30min = current_total_reward_SABER

        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        actor_buffer.append([timestep, state_buffer_actor[-1], action, reward, done_actor])

        if len(tab_state) == 0:
            for current_state in state_buffer_actor:
                tab_state.append(current_state)
        else:
            tab_state.append(state_buffer_actor[-1])
        tab_action.append(action)
        tab_reward.append(reward)
        tab_nonterminal.append(not done_actor)

        if T_actor % args.log_interval == 0:
            log(f"T = {T_actor} / {args.T_max}")
            duration_actor = time.time() - start_time_actor
            print(
                f"Time between 2 log_interval for " f"actor {id_actor} ({duration_actor:.3f} sec)"
            )
            start_time_actor = time.time()

        if T_actor % args.weight_synchro_frequency == 0:
            actor.load_weight_from_redis()

        if len(actor_buffer) >= args.length_actor_buffer:
            if (not mem_actor.transitions.actor_full) and (
                (index_actor_in_memory + len(actor_buffer)) >= mem_actor.transitions.actor_capacity
            ):
                redis_servor.set(cst.IS_FULL_ACTOR_STR + str(id_actor), 1)
                mem_actor.transitions.actor_full = True

            priorities_buffer = actor.compute_priorities(
                tab_state, tab_action, tab_reward, tab_nonterminal, mem_actor.priority_exponent
            )

            # We dont have the next_states for the last n_step states in the buffer so we just
            # set their priorities to max priorities (should be 3/args.length_buffer_actor
            # experience so a bit negligeable...)
            max_priority = np.float64(redis_servor.get(cst.MAX_PRIORITY_STR))
            last_priorities = np.ones(mem_actor.n) * max_priority

            all_priorities = np.concatenate((priorities_buffer, last_priorities))

            p = Process(
                target=send_actor_buffer,
                args=(
                    actor_buffer,
                    index_actor_in_memory,
                    id_actor,
                    mem_actor,
                    all_priorities,
                    T_actor,
                ),
            )
            p.daemon = True
            p.start()
            index_actor_in_memory = (
                index_actor_in_memory + len(actor_buffer)
            ) % args.actor_capacity
            if args.synchronize_actors_with_learner and (
                T_actor >= step_to_start_sleep
            ):  # Make actors sleep to wait learner if synchronization is on!
                # Actors are always faster than learner
                T_learner = int(redis_servor.get(cst.STEP_LEARNER_STR))
                while (
                    T_learner + 2 * args.weight_synchro_frequency <= T_actor * args.nb_actor
                ):  # We had a bug at the end because learner don't put in redis memory that
                    # he reached 50 M and actor was sleeping all time...
                    time.sleep(cst.TIME_TO_SLEEP)
                    T_learner = int(redis_servor.get(cst.STEP_LEARNER_STR))
            actor_buffer = []

            tab_state = []
            tab_action = []
            tab_reward = []
            tab_nonterminal = []

        # Update target network
        if T_actor % args.target_update == 0:
            actor.update_target_net()

        # Plot and dump in csv every evaluation_interval steps (there is in fact not any
        # evaluation done, we just keep track of score while training)
        if (
            T_actor % (args.evaluation_interval / args.nb_actor) == 0
            and id_actor == 0
            and T_actor >= (initial_T_actor + args.evaluation_interval / 2)
        ):
            pipe = redis_servor.pipeline()
            pipe.get(cst.STEP_LEARNER_STR)
            for id_actor_loop in range(args.nb_actor):
                pipe.get(cst.STEP_ACTOR_STR + str(id_actor_loop))
            step_all_agent = pipe.execute()

            T_learner = int(
                step_all_agent.pop(0)
            )  # We remove first element of the list because it's the number of learner step
            T_total_actors = 0
            if args.nb_actor == 1:  # If only one actor, we can just get this value locally
                T_total_actors = T_actor
            else:
                for nb_step_actor in step_all_agent:
                    T_total_actors += int(nb_step_actor)

            Tab_T_actors.append(T_total_actors)
            Tab_T_learner.append(T_learner)

            current_avg_episode_length = sum(episode_length_buffer) / len(episode_length_buffer)
            Tab_length_episode.append(current_avg_episode_length)

            indice_longest_episode = np.argmax(episode_length_buffer)
            Tab_longest_episode.append(
                (
                    episode_length_buffer[indice_longest_episode],
                    total_reward_buffer_SABER[indice_longest_episode],
                )
            )

            current_avg_reward = sum(total_reward_buffer_SABER) / len(total_reward_buffer_SABER)

            log(f"T = {T_total_actors} / {args.T_max} | Avg. reward: {current_avg_reward}")

            tab_rewards_plot.append(list(total_reward_buffer_SABER))

            # Plot
            _plot_line(
                Tab_T_actors,
                Tab_T_learner,
                Tab_length_episode,
                Tab_longest_episode,
                tab_rewards_plot,
                "Reward_" + args.game,
                path=args.path_to_results,
            )

            dump_in_csv(
                args.path_to_results,
                args.game,
                T_total_actors,
                T_learner,
                total_reward_buffer_5min,
                total_reward_buffer_30min,
                total_reward_buffer_SABER,
                episode_length_buffer,
            )

            for filename in os.listdir(args.path_to_results):
                if "last_model_" + args.game in filename:
                    try:
                        os.remove(os.path.join(args.path_to_results, filename))
                    except OSError:
                        print(
                            f"last_model_{args.game} were not found, "
                            f"that's not suppose to happen..."
                        )
                        pass
            actor.save(
                args.path_to_results,
                T_total_actors,
                T_learner,
                f"last_model_{args.game}_{T_total_actors}.pth",
            )

            if current_avg_reward > best_avg_reward:
                best_avg_reward = current_avg_reward
                actor.save(
                    args.path_to_results, T_total_actors, T_learner, f"best_model_{args.game}.pth"
                )

        state_buffer_actor = next_state_buffer_actor
        timestep += 1
        T_actor += 1


def main():
    args = return_args()

    redis_servor = None
    while True:
        try:
            redis_servor = redis.StrictRedis(host=args.host_redis, port=args.port_redis, db=0)
            redis_servor.set("foo", "bar")
            redis_servor.delete("foo")
            print("Connected to redis servor.")
            break

        except redis.exceptions.ConnectionError as error:
            logging.error(error)
            time.sleep(1)

    # Check if learner finished to initialize the redis-servor
    model_weight_from_learner = redis_servor.get(cst.MODEL_WEIGHT_STR)
    while model_weight_from_learner is None:
        print(
            "redis servor not initialized, probably because learner is still working on it"
        )  # This should not take more than 30 seconds!
        time.sleep(10)
        model_weight_from_learner = redis_servor.get(cst.MODEL_WEIGHT_STR)
    launch_actor(args.id_actor, args, redis_servor)


if __name__ == "__main__":
    main()
