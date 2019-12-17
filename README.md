Rainbow-IQN Ape-X :
=======

License: Apache 2.0

Rainbow-IQN Ape-X is a new distributed state-of-the-art algorithm on Atari coming
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
Learning really superhuman on Atari* [[4]](#references) on which
we introduce SABER: a **S**tandardized Atari **BE**nchmark for general 
**R**einforcement learning algorithms.<br/><br/>
Importantly this code was the Reinforcement Learning part of the algorithm 
I developed to win the 
[CARLA challenge](https://carlachallenge.org/results-challenge-2019/)  on Track 2 *Cameras Only*. This success showed the 
strength of Rainbow-IQN Ape-X as a general algorithm.


Requirements/Installation
------------

- Python 3.6+
- Pytorch >= 0.4.1
- CUDA 9.0 or higher
- [redis (link to install for Ubuntu 16)](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-redis-on-ubuntu-16-04)

To install all dependencies with Anaconda run `$ conda env create -f environment.yml`. <br/>
If no Anaconda, install [pytorch](https://pytorch.org/) and then install the following packages
with pip: atari-py, redlock-py, plotly, opencv-python.<br/>
You can take a look at the Dockerfile if you are uncertain about steps to install this project.

Afterwards, you can install the package with:
```bash
$ pip install --editable ./rainbow-iqn-apex
```

You will be able to use functions and classes from this project into other projects,
if you make changes to the sources files, those changes will be immediately
seen next time you restart the python interpreter 
(or reload the package with importlib):
```python
import rainbowiqn
```

Uninstall it with:
```bash
$ pip uninstall rainbow-iqn-apex
```


This code has been tested on Ubuntu 16 and 18. <br/>

Sanity check
------------

Open 3 terminal to sanity check if every thing is working (this will launch an experiment with one actor on space_invaders): <br/>
```bash
# Terminal 1. This launchs the redis servor on port 6379.
$ redis-server redis_rainbow_6379.conf 
 
# Terminal 2. This launchs the learner.
$ python rainbowiqn/launch_learner.py --memory-capacity 100000 \
                                      --learn-start 8000 \
                                      --log-interval 2500
                                      
# Terminal 3. This launchs the actor.
$ python rainbowiqn/launch_actor.py --id-actor 0 \
                                    --memory-capacity 100000 \
                                    --learn-start 8000 \
                                    --log-interval 2500
```
If after a short time (1 minute probably), you see some logs like the following one appearing in the learner and the actor terminal, everything is OK! <br/>
```bash
[2019-08-12T17:40:11] T = 12500 / 50000000
Time between 2 log_interval for learner (14.410 sec)  # (for the learner)

[2019-08-12T17:40:06] T = 12500 / 50000000
Time between 2 log_interval for actor 0 (13.249 sec)  # (for the actor)
```

Kill all 3 terminals after and see the [wiki](https://github.com/valeoai/rainbow-iqn-apex/wiki) to know how to launch experiments for real!

To test a pretrained snapshot, you must download trained weight from 
the [release](https://github.com/valeoai/rainbow-iqn-apex/releases) and then prompt the following command:
```bash
# Remove rendering for faster evaluation
$ python rainbowiqn/test_multiple_seed.py --model with_weight/Rainbow_IQN/space_invaders/last_model_space_invaders_50000000.pth \
                                          --game space_invaders --render
```

<a name="saber">SABER: A Standardized Atari Benchmark for general REinforcement learning algorithms</a>
------------

By default all experiments will be made on SABER. This includes all recommendations of 
*Machado et al.* [[5]](#references) (i.e. ignore life signal, using sticky actions, always use 18 action set, report 
results as the mean score over 100 consecutive training episodes) and a new parameter which we call `max_stuck_time` (5 minutes by default). <br/>
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

- This codebase is heavily borrowed from [@kaixhin](https://github.com/Kaixhin) for [Rainbow](https://github.com/Kaixhin/Rainbow) 
(see Kaixhin license there [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](Kaixhin_LICENSE.md))
- [@dopamine](https://github.com/google/dopamine) for the Tensorflow implementation 
of [IQN](https://github.com/google/dopamine/tree/master/dopamine/agents/implicit_quantile) (see `compute_loss_iqn.py` 
for Dopamine license)

<a name="ref">References</a>
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)<br/>
[2] [Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)<br/>
[3] [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)<br/>
[4] [Is Deep Reinforcement Learning really superhuman on Atari?](https://arxiv.org/abs/1908.04683)<br/>
[5] [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009)
