# Q_Learning_Basics

This is the code for my tutorials on q learning on Instagram.
The required libraries for this program are:
1. Numpy
2. Gym
3. Tqdm

In QLearning.py, to have the output of the program saved to a file, replace "output = False" with "output = True".
If you only want the summary of the program, replace

    output = False
    summary = False
to
    
    output = True
    summary = True
    
It will save to an output.txt file.

In q_learning_taxi.py and updated_q_learning_taxi.py, it will log the testing phase to a file called taxi-output.txt.

updated_q_learning_taxi.py allows you to save and load q tables.
