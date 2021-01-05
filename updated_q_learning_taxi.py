#Imports
from datetime import datetime
from os import path
import numpy as np
import time
import random
import logging
import gym


#Table saving function
def save_table(table, overwrite=False, tableId=None):
    file = None
    file_found = False
    tableId = 0
    if overwrite:
        file = f'table{tableId}'
    else:
        while not file_found:
            if not path.isfile(f'table{tableId}.npy'):
                file = f'table{tableId}'
                file_found = True
            else:
                tableId += 1
    np.save(file, table)


#Table loading function
def load_table(tableId=0, latest=False):
    if latest == True:
        file_found = False
        tableId = 0
        while not file_found:
            if not path.isfile(f'table{tableId+1}.npy'):
                file_found = True
            else:
                tableId += 1
    return np.load(f'table{tableId}.npy')


def run(train=True, table=None, episodes=1000, epsilon=0.5, decay=0.01, display=False):
    penalties = 0
    if table != None:   #Load a past table, if selected
        if table == 'latest':
            q_table = load_table(latest=True)
        elif type(table) == int:
            try:
                q_table = load_table(tableId=table)
            except:
                print('Table load failed. Halting')
                logging.error('Table load failed. Program halted')
                quit()
        else:
            print('Unknown input. Using latest')
            logging.warn('Unknown input to train function. Using latest table')
            load_table(latest=True)
    else:
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

    if train:           #For training phase, i.e. when train=True
        for i in range(episodes):
            state = env.reset()
            epochs, penalties, reward, = 0, 0, 0
            done = False
            total_reward = 0

            while not done:
                if random.uniform(0, 1) < epsilon:  # Explore
                    action = env.action_space.sample()
                else:  # Exploit
                    action = np.argmax(q_table[state])

                # Get new states, reward, check if done, and info
                next_state, reward, done, info = env.step(action)
                old_value = q_table[state, action]  # Get old q value
                # Get next maximum reward
                next_max = np.max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * \
                    (reward + gamma * next_max)
                q_table[state, action] = new_value  # Update q table

                if reward <= -10:
                    penalties += 1

                state = next_state
                epochs += 1
                total_reward += reward
        #        logging.info(str(state))
                if i%1000 == 0 and display:
                    #env.render()
                    pass

            if i % 1000 == 0:
                print('\b'*1000)
                print(
                    f'Episode {i}/{episodes} Epochs = {epochs} Total reward = {total_reward} Penalties = {penalties}', end='')
                epsilon -= decay  # Decay epsilon
    else:           #For the testing phase, i.e. when train=False
        done = False
        state = env.reset()
        epochs = 0
        total_reward = 0
        while not done:
            action = np.argmax(q_table[state])

            # Get new states, reward, check if done, and info
            next_state, reward, done, info = env.step(action)
            old_value = q_table[state, action]  # Get old q value
            # Get next maximum reward
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state, action] = new_value  # Update q table

            if reward <= -10:
                penalties += 1

            state = next_state
            epochs += 1
            total_reward += reward
            env.render()
#           logging.info(str(state))

            logging.info(str(state) + ' ' + str(reward))
        print(epochs)
    return q_table


#Setup logging. It will output to a file called taxi-output.txt in the same directory
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='taxi-output.txt'
)

logging.info('----- ----- Running ----- -----')

#Initializing all vars and environment

alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 100000
decay = 0.0001
all_epochs = []
all_penalties = []

env = gym.make('Taxi-v3').env
env.reset()

#Check if the user wants to train the AI, or just wants to see how it goes

mode = input('Train? [y/n] >>> ')
while mode not in ['y', 'n', 'Y', 'N']:
    mode = input('Train? [y/n] >>> ')

#Start training
if mode.lower() == 'y':
    q_table = run(train=True, table=None, episodes=episodes, epsilon=epsilon, decay=decay)             #If you want to load a past q table, change this parameter to it's id
    print('\nDone')

    #Save the table. Type 'n' if you dont want it saved
    save = input('Save? [y/n] >>> ')
    while mode not in ['y', 'n', 'Y', 'N']:
        mode = input('Save? [y/n] >>> ')
    if save.lower() == 'y' and mode.lower() == 'y':
        print('\nSaving file')
        save_table(q_table)
        time.sleep(1.5)

print('Displaying trained model')
run(train=False, table='latest')