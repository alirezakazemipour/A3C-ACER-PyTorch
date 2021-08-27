[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  

# A3C-ACER-PyTorch

This repository contains **PyTorch Implementation** of papers **Sample Efficient Actor-Critic with Experience Replay** (a.k.a **ACER**) and, **Asynchronous Methods for Deep Reinforcement Learning** (a.k.a. **A3C**.)

The A3C paper introduced some key ideas that can be summarized into:
1. Asynchronous updates from multiple parallel agents to decorrelates the agent's data into a more stationary process rather than maintaining an Experience Replay Memory. Consequently, exceeding limits of off-policy methods and also, reducing memory computation per real interaction with the environment.
2. Architectures that share layers between the policy and value function. Thus, providing better and more efficient representation learning and feature extraction.
3. An updating scheme that operates on fixed-length segments of experiences (say, 5 or 20 time-steps) that increase the stationarity of the agent's data.

**But:**  
A3C's lack of Experience Replay means it is considerably Sample-Inefficient and the number of interactions with the environment, needed to solve the task, is consequently **high**.

Based on this deficit of the A3C, ACER introduces an actor-critic method upon A3C's core structure accompanied by the benefits of having thread-based Experience Replays to improve sample efficiency. 
More precisely, in the ACER algorithm, each of the parallel agents, on the one hand, performs A3C-like on-policy updates, and on the other, has its own Experience Replay buffer to perform off-policy updates.

Also, the ACER utilizes more advanced techniques like Truncated Importance Sampling with Bias Correction, Stochastic Dueling Network Architectures and, Efficient Trust Region Policy Optimization to further improve stability (which is a common challenge in Policy Gradient methods) and also helps increasing Sample Efficiency even more.

**This repository contains the discrete implementation of the ACER [here](https://github.com/alirezakazemipour/A3C-ACER-PyTorch) and the A3C's [here](https://github.com/alirezakazemipour/A3C-ACER-PyTorch/tree/A3C_Atari)**.

>However, continuous implementations are also provided [here](https://github.com/alirezakazemipour/A3C-ACER-PyTorch/tree/ACER_Continuous) for the ACER and [here](https://github.com/alirezakazemipour/A3C-ACER-PyTorch/tree/A3C_Continuous) for A3C, they have not been tested yet and they will be added to the current work whenever they're suitably debugged and validated in the future.

## Results
> number of parallel agents = 8.  
> x-axis corresponds episode number.

ACER's Output| ACER's Output
:-----------------------:|:-----------------------:
![](Readme%20files/Gifs/SpaceInvaders.gif)| ![](Readme%20files/Gifs/Breakout.gif)
Running Episode Reward| Running Episode Reward
![](Readme%20files/Plots/SpaceInvaders_reward.png)| ![](Readme%20files/Plots/Breakout_reward.png) 
Running Episode Length| Running Episode Length
![](Readme%20files/Plots/SpaceInvaders_ep_len.png)| ![](Readme%20files/Plots/Breakout_ep_len.png) 


### Comparison
> number of parallel agents = 2  
>  x-axis corresponds episode number.


ACER's Output| Recurrent A3C's Output
:-----------------------:|:-----------------------:
![](Readme%20files/Gifs/PongACER.gif)| ![](Readme%20files/Gifs/PongRecurrentA3C.gif)
Running Episode Reward| Running Episode Reward
![](Readme%20files/Plots/PongACER_reward.png)| ![](Readme%20files/Plots/PongRecurrentA3C_reward.png) 
Running Episode Length| Running Episode Length
![](Readme%20files/Plots/PongACER_ep_len.png)| ![](Readme%20files/Plots/PongRecurrentA3C_ep_len.png) 


- The Sample Efficiency promised by the ACER is obvious as it can be seen on the left plot that the score of &cong; 21 has been achieved 600 episodes vs. 1.7k episodes of the Recurrent A3C on the right.

## Table of Hyperparameters
Parameter| Value
:-----------------------:|:-----------------------:|
lr			     | 1e-4
entropy coefficient | 64
gamma	          | 0.99
k (rollout length) | 20
total memory size (Aggregation of all parallel agents' replay buffers)| 6e+5
per agent replay memory size | 6e+5 // (number of agents * rollout length)
c (used in truncated importance sampling)| 10
&delta; (used in trust-region computation)| 1
replay ratio| 4
polyak average coefficients | 0.01 ( = 1 - 0.99)
critic loss coefficient| 0.5
max grad norm| 40


## Dependencies

- PyYAML == 5.4.1
- cronyo == 0.4.5
- gym == 0.17.3
- numpy == 1.19.2
- opencv_contrib_python == 4.4.0.44
- psutil == 5.5.1
- torch == 1.6.0

## Installation

```bash
pip3 install -r requirements.txt
```

## Usage

### How to Run
```bash
usage: main.py [-h] [--env_name ENV_NAME] [--interval INTERVAL] [--do_train]
               [--train_from_scratch] [--seed SEED]

Variable parameters based on the configuration of the machine or user's choice

optional arguments:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   Name of the environment.
  --interval INTERVAL   The interval specifies how often different parameters
                        should be saved and printed, counted by episodes.
  --do_train            The flag determines whether to train the agent or play with it.
  --train_from_scratch  The flag determines whether to train from scratch or continue previous tries.
  --seed SEED           The randomness' seed for torch, numpy, random & gym[env].
```
- **In order to train the agent with default arguments, execute the following command and use `--do_train` flag, otherwise the agent would be tested** (You may change the environment and random seed based on your desire.):
```shell
python3 main.py --do_train --env_name="PongNoFrameskip-v4" --interval=200
```
- **If you want to keep training your previous run, execute the following (add `--train_from_scratch` flag):**
```shell
python3 main.py --do_train --env_name="PongNoFrameskip-v4" --interval=200 --train_from_scratch
```

### Pre-Trained Weights
- There are pre-trained weights of the agents that were shown in the [Results](#Results)  section playing, if you want to test them by yourself, please do the following:
1. First extract your desired weight from `*tar.xz` format to get `.pth` extension then, rename your _env_name_ + _net_weights.pth_ file to _net_weights.pth_. For example: `Breakout_net_weights.pth` -> `net_weights.pth`
2. Create a folder named _Models_  in the root directory of the project and **make sure it is empty**.
3. Create another folder with an arbitrary name inside _Models_ folder. For example:  
```bash
mkdir Models/ Models/temp_folder
```
4. Put your `net_weights.pth` file in your _temp_folder_.
5. Run above commands without using `--do_tarin` flag:  
```shell
python3 main.py --env_name="PongNoFrameskip-v4"
```

### Hardware Requirements
- All runs with 8 parallel agents were carried out on [paperspace.com](https://www.paperspace.com/) [Free-GPU, 8 Cores, 30 GB RAM].
- All runs with 8 parallel agents were carried out on [Google Colab](https://colab.research.google.com) [CPU Runtime, 2 Cores, 12 GB RAM].

## Tested Environments
- [x] PongNoFrameskip-v4
- [x] BreakoutNoFrameskip-v4
- [x] SpaceInvadersNoFrameskip-v4
- [ ] AssaultNoFrameskip-v4

## TODOs
- [ ] Verify and add results of the Continuous version of ACER
- [ ] Verify and add results of the Continuous version of A3C

## Structure
```bash
.
├── Agent
│   ├── __init__.py
│   ├── memory.py
│   └── worker.py
├── LICENSE
├── main.py
├── NN
│   ├── __init__.py
│   ├── model.py
│   └── shared_optimizer.py
├── Pre-Trained Weights
│   ├── Breakout_net_weights.tar.xz
│   ├── PongACER_net_weights.tar.xz
│   ├── PongRecurrentA3C_net_weights.tar.xz
│   └── SpaceInvaders_net_weights.tar.xz
├── Readme files
│   ├── Gifs
│   │   ├── Breakout.gif
│   │   ├── PongACER.gif
│   │   ├── PongRecurrentA3C.gif
│   │   └── SpaceInvaders.gif
│   └── Plots
│       ├── Breakout_ep_len.png
│       ├── Breakout_reward.png
│       ├── PongACER_ep_len.png
│       ├── PongACER_reward.png
│       ├── PongRecurrentA3C_ep_len.png
│       ├── PongRecurrentA3C_reward.png
│       ├── SpaceInvaders_ep_len.png
│       └── SpaceInvaders_reward.png
├── README.md
├── requirements.txt
├── training_configs.yml
└── Utils
    ├── atari_wrappers.py
    ├── __init__.py
    ├── logger.py
    ├── play.py
    └── utils.py
```
1. _Agent_ package includes of the agent's specific configurations like its memory, thread-based functions, etc.
2. _NN_ package includes the Neural Network's structure and its optimizer settings.
3. _Utils_ includes minor codes that are common for most RL codes and do auxiliary tasks like logging, wrapping Atari environments and... .
4. _Pre-Trained Weights_ is the directory that pre-trained weights have been stored at.
5. Gifs and plot images of the current Readme file lies at the _Readme files_ directory.

## Reference
1. [_Sample Efficient Actor-Critic with Experience Replay_, Wang, et al., 2016](https://arxiv.org/abs/1611.01224)
2.  [_Asynchronous Methods for Deep Reinforcement Learning_, Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
3.  [_OpenAI Baselines: ACKTR & A2C_](https://openai.com/blog/baselines-acktr-a2c/)

## Acknowledgement
Current code was inspired by following implementation **especially the first one**:
1. [acer](https://github.com/openai/baselines/tree/master/baselines/acer) by [@OpenAI](https://github.com/openai)
2. [ACER](https://github.com/Kaixhin/ACER) by [@Kaixhin ](https://github.com/Kaixhin)
3. [acer](https://github.com/dchetelat/acer) by [@dchetelat ](https://github.com/dchetelat)
4. [ACER_tf](https://github.com/hercky/ACER_tf) by [@hercky](https://github.com/hercky)
