import argparse
import os
import random

import torch


def return_args():
    parser = argparse.ArgumentParser(description="Rainbow-IQN")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--game", type=str, default="space_invaders", help="ATARI game")
    parser.add_argument(
        "--proba-sticky-actions",
        type=float,
        default=0.25,
        help="Proba of sticky actions (see Revisiting ALE for more detail), "
        "default is 0.25, use 0 to disable",
    )
    parser.add_argument(
        "--T-max",
        type=int,
        default=int(50e6),
        metavar="STEPS",
        help="Number of training steps (this need to be multiplied by the action repeat "
        "to get the number of frames)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=4,
        metavar="T",
        help="Number of consecutive states processed",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=4,
        metavar="T",
        help="Number of time action is repeated (this is highly influenced by sticky actions!",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512, metavar="SIZE", help="Network hidden size"
    )
    parser.add_argument(
        "--noisy-std",
        type=float,
        default=0.1,
        metavar="σ",
        help="Initial standard deviation of noisy linear layers. I dont understand from "
        "the original Rainbow paper if this should be 0.5 or 0.1. Anyway 0.1 was "
        "the value used for all experiments",
    )
    parser.add_argument(
        "--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)"
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=int(1e6),
        metavar="CAPACITY",
        help="Experience replay memory capacity",
    )
    parser.add_argument(
        "--replay-frequency",
        type=int,
        default=4,
        metavar="k",
        help="Frequency of sampling from memory",
    )
    parser.add_argument(
        "--priority-exponent",
        type=float,
        default=0.2,
        metavar="ω",
        help="Prioritised experience replay exponent (originally denoted α), its 0.5 for original"
        " Rainbow but R-IQN need a lower value",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=0.4,
        metavar="β",
        help="Initial prioritised experience replay importance sampling weight",
    )
    parser.add_argument(
        "--multi-step",
        type=int,
        default=3,
        metavar="n",
        help="Number of steps for multi-step return",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, metavar="γ", help="Discount factor"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=int(32e3),
        metavar="τ",
        help="Number of steps after which to update target network",
    )
    parser.add_argument(
        "--reward-clip",
        type=int,
        default=1,
        metavar="VALUE",
        help="Reward clipping (0 to disable)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00005,
        metavar="η",
        help="Learning rate from IQN paper (different than Rainbow)",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=0.0003125,
        metavar="ε",
        help="Adam epsilon from IQN paper (different than Rainbow)",
    )
    parser.add_argument("--batch-size", type=int, default=32, metavar="SIZE", help="Batch size")
    parser.add_argument(
        "--learn-start",
        type=int,
        default=int(80e3),
        metavar="STEPS",
        help="Number of steps before starting training",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=500000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=100,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=500,
        metavar="N",
        help="Number of transitions to use for validating Q",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25000,
        metavar="STEPS",
        help="Number of training steps between logging status",
    )
    parser.add_argument("--render", action="store_true", help="Display screen")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN")

    parser.add_argument(
        "--host-redis",
        metavar="H_RED",
        default="localhost",
        help="IP of the host server for Redis database (default: localhost)",
    )
    parser.add_argument(
        "-p_red",
        "--port-redis",
        metavar="P_RED",
        default=6379,
        type=int,
        help="TCP port to listen to for Redis database (default: 6379)",
    )

    parser.add_argument(
        "--weight-synchro-frequency",
        type=int,
        default=400,
        help="Frequency of learner pushing weight and actors getting them from redis database "
        "(default: 400)",
    )
    parser.add_argument(
        "--length-actor-buffer",
        type=int,
        default=1000,
        help="Size of actor_buffer, before being pushed to the redis memory (default: 1000)",
    )
    parser.add_argument(
        "--nb-actor", type=int, default=1, help="Number of actor to use (default: 1)"
    )

    parser.add_argument(
        "--id-actor", type=int, default=-1, help="Current actor id, -1 means the learner"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=5,
        help="Multiprocessing Queue size, filled by mem.sample and read to make the backprop."
        " This is just to be more effecient (and got really few algorithmic impact",
    )

    # IQN parameters
    parser.add_argument("--kappa", default=1.0, type=float, help="kappa for Huber Loss in IQN")
    parser.add_argument(
        "--num-tau-samples", default=64, type=int, help="N in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--num-tau-prime-samples", default=64, type=int, help="N' in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--num-quantile-samples", default=32, type=int, help="K in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--quantile-embedding-dim", default=64, type=int, help="n in equation 4 in IQN paper"
    )
    # IQN parameters

    # Rainbow only parameters(no IQN and C51 instead)
    parser.add_argument(
        "--rainbow-only",
        type=int,
        default=0,
        help="Remove IQN and use standard Rainbow instead (still distributed with Ape-X",
    )
    parser.add_argument(
        "--atoms", type=int, default=51, metavar="C", help="Discretised size of value distribution"
    )
    parser.add_argument(
        "--V-min",
        type=float,
        default=-10,
        metavar="V",
        help="Minimum of value distribution support",
    )
    parser.add_argument(
        "--V-max",
        type=float,
        default=10,
        metavar="V",
        help="Maximum of value distribution support",
    )
    # Rainbow only parameters(no IQN and C51 instead)

    parser.add_argument("--gpu-number", type=int, default=0, help="GPU to use")

    parser.add_argument(
        "--disable-SABER-mode",
        action="store_true",
        help="disable SABER condition, i.e. max_length_episode to 100 hours "
        "and ending episode after 5 minutes without any reward",
    )

    parser.add_argument(
        "--synchronize-actors-with-learner",
        type=int,
        default=1,
        help="Should the actors wait for the learner (i.e. one step of learner for 4 steps of"
        "actor by default). Synchronization should be used in a single agent setting for"
        "reproducibility but we recommend to remove it if using more than 4 actors",
    )
    parser.add_argument(
        "--path-to-results",
        type=str,
        default=None,
        help="Path to the results folder to store the results file. This is also used to resume"
        "a stopped experiment (default is results folder in project root)",
    )
    # Setup
    args = parser.parse_args()

    # Handling synchronisation print
    if (not args.synchronize_actors_with_learner) and args.nb_actor == 1:
        print(
            "YOU ARE USING ONLY ONE ACTOR BUT DONT SYNCHRONISE IT WITH THE LEARNER, THIS IS "
            "PROBABLY NOT WHAT YOU WANT, FOR REPRODUCIBILITY PURPOSE, LEARNER SHOULD BE "
            "SYNCHRONISED WITH ACTOR (one learner step each 4 actor step by default"
        )
        raise Exception  # Comment this line if it's really what you want to do

    if args.synchronize_actors_with_learner and args.nb_actor > 4:
        print(
            "YOU ARE USING MORE THAN 4 ACTORS AND STILL SYNCHRONISE THEM WITH THE LEARNER,"
            "THIS IS PROBABLY NOT WHAT YOU WANT, THEY WILL WAIT A LOT DOING NOTHING"
        )
        raise Exception  # Comment this line if it's really what you want to do

    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + f"{k}: {v}")

    if args.path_to_results is None:
        args.path_to_results = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "results"
        )

    # If no model gave as argument, we check if there is an experiment to resume from.
    # If there is a file name last_model_name_game in the --path-to-results folder
    # then we resume the stopped experiment with this snapshot.
    # If --model is given we just start experiment from scratch with weight initialize with --model
    # This allow to resume experiment by default (with the exact same command to launch or resume)
    if args.model is None:
        last_model = None
        for filename in os.listdir(args.path_to_results):
            if "last_model_" + args.game in filename:
                last_model = os.path.join(args.path_to_results, filename)
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                checkpoint = torch.load(last_model, map_location="cpu")
                step_actors_already_done = checkpoint["T_actors"]
                step_learner_already_done = checkpoint["T_learner"]
                break
        if last_model:
            args.continue_experiment = True
            print(
                "We are restarting a stopped experiment, we have to check now for the last model"
            )

            print(
                f"We found the filename {last_model} to restart, number of step actor already "
                f"done is {step_actors_already_done}"
            )
            print(
                "We found the filename "
                + last_model
                + " to restart, number of step learner already"
                " done is ",
                step_learner_already_done,
            )

            args.step_actors_already_done = step_actors_already_done
            args.step_learner_already_done = step_learner_already_done
            args.model = last_model
        else:
            args.continue_experiment = False
    else:
        args.continue_experiment = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    args.seed = args.seed + args.id_actor
    print("We take a different random seed for each actors/learner")
    args.actor_capacity = args.memory_capacity // args.nb_actor
    print("we set actor_capacity to ", args.actor_capacity)

    if not args.disable_SABER_mode:
        print(
            "WE USE SABER MODE, so we put max length episode to 100 hours and will end episode"
            " when stuck for 5 minutes, i.e. 5 minutes without any reward"
        )
        args.max_episode_length = 100 * 3600 * 60  # 100 hours (at 60 Hz)
        args.max_frame_stuck_SABER = 5 * 60 * 60  # 5 minutes (at 60 Hz)
    else:
        print("THIS IS NOT ADVISED, let's set a max_episode_length of 30 minutes")
        # We use 30 minutes when no SABER mode (like Rainbow, DQN, IQN...).
        args.max_episode_length = 30 * 60 * 60

    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device("cuda")
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = not args.disable_cudnn
    else:
        args.device = torch.device("cpu")

    if args.rainbow_only:
        print("Launching an experiment with Rainbow only (i.e. C51 instead of IQN)")
    else:
        print(
            "Launching an experiment with Rainbow IQN (you can disable"
            " IQN by using --rainbow-only True)"
        )

    return args
