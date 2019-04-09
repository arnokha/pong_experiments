import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
import sys
import time

# hyperparameters
H = 30 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
save_counter = 0
total_reward = 0

render=False
plotting=False

D = 1 ## where we are - where we need to go
n_actions = 3

## Model Initialization
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H,n_actions) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    #I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    #I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I[:-1,:,0]
    return I.astype(np.float)

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(h, model['W2'])
    p = stable_softmax(logp)
    return p, h # return probability of taking actions, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()    
    dh = np.dot(epdlogp, model['W2'].T)
    dh[eph <= 0] = 0 # backprop relu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

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
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

## Actions
action_up = 2
action_down = 5
action_nop = 0
actions = [action_up, action_down, action_nop]

observation = env.reset()

steps=0
ys = []
paddle_height = 10
start = time.time()
prev_paddle_y = -1
target_loc = 55
vel = 0
#for i in range (1500):
while(episode_number < 2000):
    if render: env.render()

    # preprocess the observation
    cur_x = prepro(observation)
    paddle_y = get_paddle_y(cur_x)

    #if paddle_y != -1:
    if paddle_y != -1 and prev_paddle_y != -1:
        vel = paddle_y - prev_paddle_y
        x = np.array([target_loc - paddle_y])
    else:
        vel = 0
        x = np.zeros(D)

    # forward the policy network and sample an action from the returned probability
    aprobs, h = policy_forward(x)
    action_idx = np.random.choice(n_actions, p=aprobs)
    action = actions[action_idx]

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    #aprobs.append(aprob)
    #y = 1 if action == action_up else 0 # a "fake label"
    y = np.zeros(n_actions)
    y[action_idx] = 1 # action taken
    ys.append(y)
    #dscore = aprobs
    #dscore[action_idx] -= 1
    #print(aprobs - y)
    #dlogps.append(aprobs - y) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprobs) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    #dlogps.append(y - aprobs) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    steps += 1

    ## ~~~~~~~~~~~~~~~~~~
    ## Reward Assignment
    ## ~~~~~~~~~~~~~~~~~~
    if paddle_y == -1:
        reward = 0
        no_op_counter = 0
    elif np.abs(x[0]) < (paddle_height / 2) and vel == 0:
        #print("reward achieved")
        reward = 2.5
        target_loc = int(np.random.random() * 65 + 20)
    elif action == action_up or action == action_up:
        #print("moved and incurred penalty")
        reward = -.01
    else:
        reward = -.01 / 2

    reward_sum += reward
    prev_paddle_y = paddle_y

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    if done: # an episode finished
        #print("Total reward for this ep({0:d}): ".format(episode_number) + str(reward_sum))
        print(episode_number, reward_sum)
        episode_number += 1
        #print("This epsiode lasted " + str(steps) + " steps")
        steps = 0

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        #epdlogp *= epr # modulate the gradient with advantage (PG magic happens right here.)

        grad = policy_backward(eph, epdlogp)
        grad["W2"] = grad["W2"].reshape(-1,n_actions)
        #print(grad)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            #p1,_ = policy_forward(np.array([1]))
            #n1,_ = policy_forward(np.array([-1]))
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                #model[k] += learning_rate * g
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            #p2,_ = policy_forward(np.array([1]))
            #n2,_ = policy_forward(np.array([-1]))
            #if p2 >= p1 and n2 <= n1:
            #    print("Policy is getting better, or staying the same")
            #else:
            #    print("Policy is getting worse :(")

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        reward_sum = 0
        if episode_number % 100 == 0:
            pickle.dump(model, open('models/control_save_h'+ str(H) +'_' + str(save_counter) + '.p', 'wb'))
            save_counter +=1
        observation = env.reset() # reset env
        prev_x = None

    #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    #    print(episode_number, reward)


end = time.time()
print(end - start)
