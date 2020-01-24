# dice-fraud-detection

This software implements a machine-learning algorithm to detect whether a Dungeons and Dragons 5th edition character has has its statistics using the formula as described in the rules. 

The Player's Handbook specifies that, to generate a single statistic value, one should roll four six-sided dice, delete the lowest one, and sum up the rest. This is often summarised by the following shorthand, 4d6dl1 ("four d6 drop lowest 1"). 

Since such dice rolls are easy to simulate on a computer, we can generate real and faked data, by using other generation formulae or algorithms, and perform a logistic regression using a trained Neural Network to estimate whether the data fits expected statistics or not. 
