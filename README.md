# dice-fraud-detection

This software implements a machine-learning algorithm to detect whether a Dungeons and Dragons 5th edition character has has its statistics using the formula as described in the rules. 

The Player's Handbook specifies that, to generate a single statistic value, one should roll four six-sided dice, delete the lowest one, and sum up the rest. This is often summarised by the following shorthand, 4d6dl1 ("four d6 drop lowest 1"). 

Since such dice rolls are easy to simulate on a computer, we can train a Neural Network to recognise their statistical features. Specifically, were another player to try and cheat their team-mates and the Dungeon Master by using an appropriately tweaked generation method, with higher average or lower standard deviation for instance, the NN would recognise abnormal features and reject it.

In theory, a well-motivated enough probabilist could compute by hand the analytical expressions for a variant of the T-test (which aims to detect if a small number of sample points were generated using a normal distribution) based around discrete uniform distributions, but the point is to illustrate the power of Neural Networks for fraud detection on a very controllable example. Dungeons and Dragons dice rolls are at the same time practical and concrete enough for this project to feel like a fun, realistic endeavour, and also inherently easy to generate in very large quantities on a computer.

The code runs in three steps:

* The first script, data-generation.py, provides real and faked datasets as output. The "raw" files contained untreated datasets, classified by method, making it possible to observe their statistics. The "Balanced" set is trimmed, shuffled and labelled real/fake, in view of treatment via a Neural Network.

* The second script, nn-training.py, takes a generated dataset and trains a simple Neural Network to recognise which datapoints are the real and faked ones. It then exports it for the end-user.

* The third script, nn-predict.py, loads a saved trained Neural Network structure and predicts whether an inputted stat block was legitimately generated or not. 
