"""
Simple Controller module that attempts to navigate to randomly generated target locations
 - Reward based on getting within 7 pixels of target location and having zero velocity
 - Use paddle velocity as input to the reinforcement learning algo as well
 
Acknowledgments:
Largely adapted from Andrej Karpathy's pong playing agent and [this article](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf)
 
 """

import argparse
import warnings
import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
import sys
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

## Default values
default_h1 = 20
default_max_episodes = 1000

## Parse arguments
parser = argparse.ArgumentParser(description='Run a policy with h hidden units')
parser.add_argument('hidden_layer_units', metavar='hidden_layer_units', type=int, nargs='?',
                    help='The number of hidden units to use in the hidden layer')
parser.add_argument('episodes_to_run', metavar='episodes_to_run', type=int, nargs='?',
                    help='The number of episodes to run the training algorithm for')
args = parser.parse_args()

if vars(args)["hidden_layer_units"] is not None:
    h1 = vars(args)["hidden_layer_units"]
else:
    #warnings.warn("Using default number of hidden units: {}".format(default_h1))
    h1 = default_h1

if vars(args)["episodes_to_run"] is not None:
    max_episodes = vars(args)["episodes_to_run"]
else:
    #warnings.warn("Using default number of hidden units: {}".format(default_h1))
    max_episodes = default_max_episodes
    
## CUDA
device=None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

## Policy
class Policy(nn.Module):
    def __init__(self, D_in, h1=128):
        super(Policy, self).__init__()
        #self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        
        self.l1 = nn.Linear(D_in, h1, bias=False)
        self.l2 = nn.Linear(h1, self.action_space, bias=False)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()).to(device=device) 
        self.reward_episode = []
        
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            #nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards).to(device=device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)).to(device=device)
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #torch.nn.utils.clip_grad_norm_(policy.parameters(), 2)
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []
    
def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor).to(device=device)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history.to(device=device), 
                                           c.log_prob(action).view(1).to(device=device)]).to(device=device)
    else:
        policy.policy_history = (c.log_prob(action)).to(device=device)
    return action

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    #I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    #I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I[:-1,:,0]
    return I.astype(np.float)

def get_paddle_y(img, display_message=False):
    paddle_2_x = 139 # Leftmost position of paddle 2
    paddle_height = 15

    paddle_1_color = 213
    paddle_2_color = 92
    ball_color = 236

    ## In the beginning of the game, the paddle on the left and the ball are not yet present
    not_all_present = np.where(img == paddle_2_color)[0].size == 0
    if (not_all_present):
        if display_message:
            print("One or more of the objects is missing, returning an empty list of positions")
            print("(This happens at the first few steps of the game)")
        return -1

    paddle_2_top = np.unique(np.where(img == paddle_2_color)[0])[0]
    paddle_2_bot = paddle_2_top + paddle_height

    return (paddle_2_top + paddle_2_bot) / 2

env = gym.make("Pong-v0")

## Hyperparameters
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
# h1 is a command line argument
batch_size = 10 # every how many episodes to do a param update?

D_in = 2 ## 1. (where we are - where we need to go), 2. (paddle center last frame - paddle center this frame)

policy = Policy(D_in, h1=h1)
policy = policy.to(device=device)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

resume = False # resume from previous checkpoint?
save_counter = 0
total_reward = 0
paddle_height = 15

render=False
plotting=False

if resume:
    LOAD_PATH="models/control_save_vel2_h100"
    checkpoint = torch.load(LOAD_PATH)
    policy.load_state_dict(checkpoint['model_state_dict'])
    episode_number = checkpoint['episode_number']

observation = env.reset()

steps=0
prev_x = None # used in computing the difference frame
running_reward = None
reward_sum = 0
episode_number = 0
prev_paddle_y = -1
target_loc = 55
up_down_counter = 0
no_op_counter = 0

start = time.time()
while(episode_number < max_episodes):
    if render: 
        env.render()
        time.sleep(0.5)

    # preprocess the observation
    curr_img = prepro(observation)
    paddle_y = get_paddle_y(curr_img)

    #if paddle_y != -1:
    if paddle_y != -1 and prev_paddle_y != -1:
        #x = np.array([target_loc - paddle_y])
        vel = paddle_y - prev_paddle_y
        x = np.array([target_loc - paddle_y, vel])
    else:
        vel = 0
        x = np.zeros(D_in)

    # forward the policy network and sample an action from the returned probability
    #aprobs, h = policy(x)
    action = select_action(x)
    observation, reward, done, info = env.step(action)
    steps += 1
    
    ## ~~~~~~~~~~~~~~~~~~
    ## Reward Assignment
    ## ~~~~~~~~~~~~~~~~~~
    if paddle_y == -1:
        reward = 0
        #no_op_counter = 0
    elif np.abs(x[0]) < (paddle_height / 2) and vel == 0:
        #print("reward achieved")
        reward = 2.5
        target_loc = int(np.random.random() * 100 + 20)
        #print(target_loc)
    else: # punish no-ops less
        reward = -.01

    policy.reward_episode.append(reward)
    prev_paddle_y = paddle_y
    reward_sum += reward
    
    if done: # an episode finished
        #print("Total reward for this ep({0:d}): {1:.2f}".format(episode_number, reward_sum))
        print("{0:d} {1:.2f}".format(episode_number, reward_sum))
        episode_number += 1
        #print("This epsiode lasted " + str(steps) + " steps")
        steps = 0
        
        if episode_number % batch_size == 0:
            update_policy()
        
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        reward_sum = 0
        if episode_number % 100 == 0:
            PATH = 'models/control_save_vel2_h__2layer'
            torch.save({
                'episode_number': episode_number,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
            #pickle.dump(model, open('models/control_save_vel2_h'+ str(H) +'_' + str(save_counter) + '.p', 'wb'))
            #save_counter +=1
            
        observation = env.reset() # reset env
        prev_x = None
        
end = time.time()
print(end - start)
