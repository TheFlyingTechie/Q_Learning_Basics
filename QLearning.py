import gym
import time
import numpy as np
from tqdm import tqdm

env = gym.make('MountainCar-v0')
env.reset()

learning_rate = 0.1
discount = 0.95
epochs = 25000
epsilon = 0.5
start_decay = 1
end_decay = epochs // 2
decay = epsilon/(end_decay - start_decay)
output = False
summary = False
file = open('output.txt', 'w')

show_every = 100

discrete_obs_size = [20,20]# * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / discrete_obs_size

q_table = np.random.uniform(low=-2, high=0, size=(discrete_obs_size + [env.action_space.n]))

def getDiscreteState(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))


for epoch in tqdm(range(epochs)):
    goal_reached = False
    start_time = time.time()
    if epoch%show_every == 0:
#        print(epoch)
        render = True
    else:
        render = False
    
    discrete_state = getDiscreteState(env.reset())
    done = False

    while not done:
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
            if output and not summary:
                file.write('Getting values from q table')
        else:
            action = np.random.randint(0, env.action_space.n)
            if output:
                file.write('Doing random action')
        new_state, reward, done, _ = env.step(action)
        if output and not summary:
            file.write('New state: ' + str(new_state) + ' Reward: ' + str(reward) + ' Done: ' + str(done))
        
        new_discrete_state = getDiscreteState(new_state)
        if output and not summary:
            file.write('New discrete state: ' + str(new_discrete_state))
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state+(action,)] = new_q
            if output and not summary:
                file.write('New q: ' + str(new_q))
            
        elif new_state[0] >= env.goal_position:
#            print(f'Goal reached at epoch {epoch}')
            q_table[discrete_state + (action, )] = 0
            goal_reached = True
            
        discrete_state = new_discrete_state
        
    if end_decay >= epoch >= start_decay:
        epsilon -= decay
        if not summary and output:
            file.write('Epsilon: ' + str(epsilon))
        
    if output:
        file.write('Epoch: ' + str(epoch) + ' Goal reached: ' + str(goal_reached) +\
                   ' Time for epoch: ' + str(time.time()-start_time))
    
    goal_reached = False
        
    env.close()
file.close()
