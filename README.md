Rainbow-IQN Ape-X :
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Rainbow-IQN Ape-X is a new state-of-the-art algorithm on Atari coming
from the combination of the 3 following papers:<br/>
Rainbow: Combining Improvements in Deep Reinforcement Learning [[1]](#references).<br/>
IQN: Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning [[2]](#references).<br/>
Ape-X: Distributed Prioritized Experience Replay [[3]](#references).<br/>

Introduction
------------
This repository is an open-source implementation of a distributed version of Rainbow-IQN
following Ape-X paper for the distributed part (there is also a distributed version 
of Rainbow only, i.e. Rainbow Ape-X).<br/>
The code presented here is at the basis of our paper *Is Deep Reinforcement 
Learning really superhuman on Atari* on which
we introduce SABER: a **S**tandardized Atari **BE**nchmark for general 
**R**einforcement learning algorithms.<br/>
Importantly this code was the Reinforcement Learning part of the algorithm 
I developed to win the 
[CARLA challenge](https://carlachallenge.org/results-challenge-2019/)  on Track 2 *Cameras Only*. This success showed the 
strength of Rainbow-IQN Ape-X as a general algorithm.


Requirements/Installation
------------

- Python 3.5+
- Pytorch >= 0.4.1
- CUDA 9.0 or higher
- [redis (link to install for Ubuntu 16)](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-redis-on-ubuntu-16-04)

To install all dependencies with Anaconda run `$ conda env create -f environment.yml`. <br/>
If no Anaconda, install [pytorch](https://pytorch.org/) and then install the following packages
 with pip: atari-py, redlock-py, plotly, opencv-python

This code has been tested on Ubuntu 16 and 18. <br/>

Open 3 terminal to sanity check if every thing is working (launchs an experiment with one actor on space_invaders): <br/>
`$ redis-server redis_rainbow_6379.conf`: This launchs the redis servor on port 6379.<br/>
`$ python launch_learner.py --memory-capacity 100000 --learn-start 8000 --log-interval 2500`: This launchs the learner <br/>
`$ python launch_actor.py --id-actor 0 --memory-capacity 100000 --learn-start 8000 --log-interval 2500`: This launchs the actor <br/>

If after a short time (1 minute probably), you see some logs like the following one appearing in the learner and the actor terminal, everything is OK! <br/>
`[2019-08-12T17:40:11] T = 12500 / 50000000` <br/>
`Time between 2 log_interval for learner (14.410 sec)` (for the learner) <br/>
`[2019-08-12T17:40:06] T = 12500 / 50000000` <br/>
`Time between 2 log_interval for actor 0 (13.249 sec)` (for the actor) <br/>

Kill all 3 terminals after and see next sections to know how to launch experiments for real!

Summary
------------


##### [Single actor setting](#singleActor)
1) Launching an experiment
2) Changing parameter
3) Results : logs and snapshots of model
4) Testing a trained model
5) Resuming a stopped experiment

##### [Multi-actor and/or parallel machine setting](#multiActor)
1) Removing synchronisation actors/learner
2) Launching a multi actor experiment on a multi-gpu machine
3) Launching a multi actor experiment on parallel machine

##### [SABER: A Standardized Atari Benchmark for general REinforcement learning algorithms](#saber)

##### [Acknowledgements](#acknow)

##### [References](#ref)

<a name="singleActor">Single actor setting </a>
------------

On this part we will describe how to launch experiment with only one actor. This was used for our paper to make fair 
comparaison with the single agent results of Rainbow, IQN and DQN.
#### 1) Launching an experiment
The easiest way to launch an experiment is to run the command <br/>
`$ bash start_learning_atari_virtual_env.sh` (or 
`$ bash start_learning_atari.sh` if installed manually without virtual env). <br/><br/>
By default this will launch an experiment on space_invaders for 200M frames. The script will open 3 terminals, the first 
one will be the redis-server, the second one the learner and the last one the single actor (this can be done manually
 but it's more cumbersome).

#### 2) Changing parameter

All parameters can be seen in the file `args.py`. To change parameter either add them in the bash script (by default 
only the game name, the number of actor and the gpu repartition can be changed in the bash script) or change directly 
the default value in args.py

#### 3) Results : logs and snapshots of model

All results file can be found in the `results` folder. <br/>
`Reward_#game_name#.html`: A html plot of the reward along the training steps 
(should be multiplied by the action repeat to get the number of frames) <br/>
`#game_name#.csv`: A csv file on which all score of all episodes encountered while training are dumped.
`last_model_#game_name#_#number_of_steps".pth`: The last model of the experiment<br/>
`best_model_#game_name#".pth`: The best model encountered while training
 (we average evaluation over 100 consecutives while training, see SABER for more information). Remind that taking 
 the best snapshot involve a bias and it should not be used to report results. 

#### 4) Testing a trained model

To evaluate a snapshot, just use the command <br/>
`$ python test_multiple_seed.py --game #game_name# --model #path_to_saved_snapshot#`<br/>
This will evaluate the snapshot for 100 episodes varying the random seed. Add the option `--render` 
to actually see the AI playing! (but it's way slower). <br/>
Some pretrained snapshot will be released soon (along with logs and plot file).

#### 5) Resuming a stopped experiment

Stopped experiments are resumed by default. Indeed, when an experiment is stopped and resume, we first check if there is 
a snapshot with name containing the string  "last_model_#name_game#" (see in `args.py` for more details). <br/>
If a snapshot is found, then we first fill the whole replay memory without learning and then when the memory
 is full we actually resume the learning at the steps where the snapshot was stopped. This was the fairest resume we 
 could do without having to save the whole replay memory (which is by default of 7GB...). <br/>
If you don't want to resume and actually make another experiment on the same game from scratch you should move the 
results file away from the `results` folder.


<a name="multiActor">Multi-actor and/or parallel machines setting</a>
------------

On this part, we will describe the specificity on how to run a multi-actor experiment. We will also detail 
how to run on parallel machine. All the points above (testing snapshot, resuming experiment...) apply too for 
the multi-actor setting.


#### 1) Removing synchronisation actors/learner

By default, actors and learner are synchronized: 1 step of learner each 4 steps of actors as in Rainbow, IQN, DQN etc...<br/>
If you use more than 4 actors (i.e. more than 4 instance of Atari in parallel), we recommend to remove the 
synchronisation (set the parameter `--synchronize-actors-with-learner` to `False` `in args.py`).<br/>
Indeed, if you keep synchronization, all actors will probably wait most of time doing nothing because learner will run way 
slower than actors.

#### 2) Launching a multi actor experiment on a multi-gpu machine

You can modify the number of actors in the bash script `$ bash start_learning_atari_virtual_env.sh` (or 
`$ bash start_learning_atari.sh` if installed manually without virtual env). You must also specifically indicate on which gpu 
will run each actors and the learner. <br/>
Then, running this bash script will open one terminal for each actors, one for the learner
and one for the redis-servor (e.g. 5 terminals if 3 actors).

#### 3) Launching a multi actor experiment on parallel machines

If launching on parallel machine everything must be launched one terminal by one. <br/>
Let's take the following example, launching 6 actors on Machine 0 with GPU 0/1 and learner on Machine 1 with GPU 0:<br/>
<br/>
-run the redis-server on Machine 0 with command `$ redis-server redis_rainbow_6379.conf` (we recommend to run the redis
 servor on the same physical machine than the learner even if it's not mandatory)<br/>
-run the learner on Machine 0 with command `$ python launch_learner --nb-actor 6 --synchronize-actors-with-learner False`<br/>
-run actor 0 on Machine 1 on gpu 0 with command `$ python launch_actor --nb-actor 6 --nb-actor 0 --synchronize-actors-with-learner False`<br/>
-same for actor 1 and 2 on Machine 1 on gpu 0 (changing the id to 1 and 2 respectively)<br/>
-run actor 3 on Machine 1 on gpu 1 with command `$ python launch_actor --gpu-number 1 --nb-actor 6 --nb-actor 0 --synchronize-actors-with-learner False`<br/>
-same for actor 4 and 5 on Machine 1 on gpu 1 (changing the id to 4 and 5 respectively)<br/>

`ERROR:root:Error 111 connecting to localhost:6379. Connection refused` => To allow actors (on Machine 1) to connect to the redis-server, you must add the ip of Machine 0 in 
`redis_rainbow_6379.conf` and change the parameter `--host-redis` to the IP of Machine 0 (this is relative to Redis connection).

<a name="saber">SABER: A Standardized Atari Benchmark for general REinforcement learning algorithms</a>
------------

By default all experiments will be made on SABER. This includes all recommendations of 
*Machado et al.* [[5]](#references) and a new parameter which we call `max_stuck_time` (5 minutes by default). <br/>
This parameter allows to set infinite episode and still terminate episode when agent is stuck. More details can be 
found in our paper *Is Deep Reinforcement 
Learning really superhuman on Atari* [[4]](#references).<br/>
In our paper we discuss how setting infinite episode is really important to allow for fair and comparable results.
Moreover this allows to compare against the human world record and shows the incredibly high
gap remaining before claiming of *superhuman performances*. <br/>
We showed that the use of *superhuman performances*
in previous papers is indeed misleading. General RL agents are definitely far from superhuman on most Atari games!



<a name="acknow">Acknowledgements</a>
----------------

- [@kaixhin](https://github.com/Kaixhin) for [Rainbow](https://github.com/Kaixhin/Rainbow)
- [@dopamine](https://github.com/google/dopamine) for [IQN](https://github.com/google/dopamine/tree/master/dopamine/agents/implicit_quantile)

<a name="ref">References</a>
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)<br/>
[2] [Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)<br/>
[3] [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)<br/>
[4] [Is Deep Reinforcement Learning really superhuman on Atari?](TODO)<br/>
[5] [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009)