import logging
import random
import time

import numpy as np
import redis
from torch.multiprocessing import Process

import rainbowiqn.constants as cst
from rainbowiqn.actor import Actor
from rainbowiqn.args import return_args
from rainbowiqn.env import Env
from rainbowiqn.redis_memory import ReplayRedisMemory
from rainbowiqn.utils import dump_in_csv_and_plot_reward, log
from rainbowiqn.reward_buffer import RewardBuffer


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

    actor = Actor(args, env_actor.action_space(), redis_servor)

    done_actor = True

    tab_state = []
    tab_action = []
    tab_reward = []
    tab_nonterminal = []

    # We want to warn the user when the agent reachs 100 hours of gameplay continuously improving
    # score. On thoses game the agent is superhuman (and learning should be stop maybe?)
    if not args.disable_SABER_mode:  # SABER mode: length episode can be infinite (100 hours)
        step_100_hours = int(args.max_episode_length / args.action_repeat) - 1

    if id_actor == 0:
        reward_buffer = RewardBuffer(args.evaluation_episodes, args.action_repeat)

    while T_actor < (args.T_max / args.nb_actor):
        if done_actor:
            if not args.disable_SABER_mode and timestep >= step_100_hours:
                print("Agent reachs 100 hours of gameplay while continuously improving score!"
                      "Agent is superhuman (happened only on Atlantis, Defender and Asteroids)."
                      "Learning could be stopped now...")
            if id_actor == 0 and T_actor > initial_T_actor:
                reward_buffer.update_score_episode_buffer(timestep)
            timestep = 0
            state_buffer_actor = env_actor.reset()
            done_actor = False

        if T_actor % args.replay_frequency == 0:
            actor.reset_noise()  # Draw a new set of noisy weights

        if T_actor < args.learn_start / args.nb_actor:
            # Do random actions before learning start
            action = random.randint(0, env_actor.action_space() - 1)
        else:
            # Choose an action greedily (with noisy weights)
            action = actor.act(state_buffer_actor)

        next_state_buffer_actor, reward, done_actor = env_actor.step(action)  # Step
        if args.render and id_actor == 0:
            env_actor.render()

        if id_actor == 0:
            reward_buffer.update_current_reward_buffer(timestep, reward)

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
            print(f"Time between 2 log_interval for actor {id_actor} ({duration_actor:.3f} sec)")
            start_time_actor = time.time()

        if T_actor % args.weight_synchro_frequency == 0:
            actor.load_weight_from_redis()

        # We want to send actor buffer in the redis memory with right initialized priorities
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
                target=mem_actor.transitions.append_actor_buffer,
                args=(actor_buffer, index_actor_in_memory, id_actor, all_priorities, T_actor),
            )
            p.daemon = True
            p.start()
            index_actor_in_memory = (
                index_actor_in_memory + len(actor_buffer)
            ) % args.actor_capacity
            # Make actors sleep to wait learner if synchronization is on!
            if args.synchronize_actors_with_learner and (T_actor >= step_to_start_sleep):
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
            dump_in_csv_and_plot_reward(redis_servor, args, T_actor, reward_buffer, actor)

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
