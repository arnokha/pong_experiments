""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import matplotlib.pyplot as plt
import sys

# Take command line input for number of hidden units to use
if len(sys.argv) == 1:
    H = 100
    print("Using default number of hidden units: " + str(H))
else:
    H = int(sys.argv[1])
    print("Using " + str(H) + " hidden units")


# hyperparameters
H = 100 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
save_counter = 0

render = True
plotting = False

## Model initialization

# Input dimensionality: 3 relative object positions to ball for each paddle: (horizontal position, top position, bottom position)
#                       and the difference of frames in the x and y locations of the ball
D = 3 + 3 + 2

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    #I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    #I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I[:-1,:,0]
    return I.astype(np.float)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

def get_object_positions(img, display_message=False):
    """ 
    Hacky way to get object positions. 
    We take advantage of the fact that we know the colors of each object
    We will treat the horizontal direction as the x axis, and the vertical direction as the y axis
    
    Input: image observed at current step in environment
    Output: 1d array representing object positions as follows: 
         - paddle 1 horizontal axis position
         - paddle 1 top vertical position
         - paddle 1 bottom vertical position
         - paddle 2 horizontal axis position
         - paddle 2 top vertical position
         - paddle 2 bottom vertical position
         - ball horizontal axis position (top pixel location selected)
         - ball vertical axis position (leftmost pixel location selected)
    """
    #paddle_1_cols = range(15,20)
    #paddle_2_cols = range(139,144)
    if img is None:
        return []
    
    paddle_height = 15
    paddle_1_x = 19  # Rightmost position of paddle 1
    paddle_2_x = 139 # Leftmost position of paddle 2
    
    paddle_1_color = 213
    paddle_2_color = 92
    ball_color = 236
    
    ## In the beginning of the game, the paddle on the left and the ball are not yet present
    not_all_present = np.where(img == paddle_1_color)[0].size == 0 or \
                      np.where(img == paddle_2_color)[0].size == 0 or \
                      np.where(img == ball_color)[0].size == 0 
    if (not_all_present):
        if display_message:
            print("One or more of the objects is missing, returning an empty list of positions")
            print("(This happens at the first few steps of the game)")
        return []
    
    paddle_1_top = np.unique(np.where(img == paddle_1_color)[0])[0]
    paddle_1_bot = paddle_1_top + paddle_height
    paddle_2_top = np.unique(np.where(img == paddle_2_color)[0])[0]
    paddle_2_bot = paddle_2_top + paddle_height
    
    ball_pos_all = np.where(img == ball_color)
    ball_pos_x = ball_pos_all[1][0]
    ball_pos_y = ball_pos_all[0][0]
    
    return [paddle_1_x - ball_pos_x, paddle_1_top - ball_pos_y, paddle_1_bot - ball_pos_y, 
            paddle_2_x - ball_pos_x, paddle_2_top - ball_pos_y, paddle_2_bot - ball_pos_y,
            ball_pos_x, ball_pos_y] 

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

## Indexes into position array
p1x = 0; p1y1 = 1; p1y2 = 2; 
p2x = 3; p2y1 = 4; p2y2 = 5
bx = 6; by = 7; bvx = 8; bvy = 9


while True:
    if render: env.render()

    # preprocess the observation
    cur_x = prepro(observation)
    
    positions = get_object_positions(cur_x)
    prev_positions = get_object_positions(prev_x)
    if prev_x is not None and len(prev_positions) != 0 and len(positions) != 0:
        dof_ball_x = positions[bx] - prev_positions[bx]
        dof_ball_y = positions[by] - prev_positions[by]
        x = np.array(positions[:-2] + [dof_ball_x, dof_ball_y])
    else:
        x = np.zeros(D)
    
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = action_up if np.random.uniform() < aprob else action_down # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == action_up else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        #print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        reward_sum = 0
        if episode_number % 200 == 0:
            pickle.dump(model, open('models/relative_save_h'+ str(H) +'_' + str(save_counter) + '.p', 'wb'))
            save_counter +=1
        observation = env.reset() # reset env
        prev_x = None

        if reward == 1: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
