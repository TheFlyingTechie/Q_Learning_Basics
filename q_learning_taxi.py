#Imports
import numpy as np
import random
import logging
import gym

#Setup logging. It will output to a file called taxi-output.txt in the same directory
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='taxi-output.txt'
)

#Initializing all vars and environment

alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 100000
decay = epsilon / (episodes / 1000)

all_epochs = []
all_penalties = []

env = gym.make('Taxi-v3').env
env.reset()

q_table = np.zeros([env.observation_space.n, env.action_space.n])

#Start training

for i in range(episodes):
    state = env.reset()
    epochs, penalties, reward, =0,0,0
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0,1) < epsilon:                           #Explore
            action = env.action_space.sample()
        else:                                                       #Exploit
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)           #Get new states, reward, check if done, and info
        old_value = q_table[state, action]                          #Get old q value
        next_max = np.max(q_table[next_state])                      #Get next maximum reward
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value                          #Update q table

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        total_reward += reward
#        logging.info(str(state))
    
    if i%100 == 0:
        print('\b'*1000)
        print(f'Episode {i}/{episodes} Epochs = {epochs} Total reward = {total_reward}', end='')
        epsilon -= decay                                            #Decay epsilon

print('\nDone')
print('Displaying trained model')

for i in range(2):
    state = env.reset()
    done = False
    epochs = 0

    while not done:
        print('\n', epochs)
        action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        env.render()
        logging.info(str(state))
