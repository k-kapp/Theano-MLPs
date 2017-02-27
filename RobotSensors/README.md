#RobotControl

The dataset consists of sensor measurements on a robot inside a room enclosed by walls. The robot is supposed to "follow the wall" (apparently a relatively common task in robot AI tasks), which means that it must try to keep the wall to its left as it circles the room. The task is to learn the direction the robot should be travelling in in order to move around in this manner, based on sensor information. It is a classification task.

In the original dataset, there are 3 sets: One with 24, 4, and 2 features. The dataset that has 24 features contain the original sensor information from the robot. The other two have preprocessed information, making the classification task easier. The original dataset may be found at the UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data. Information on the original paper submitted for this task is also at that link.

The code to train the robot is in the file 'ffnn.py'. It is an ordinary feedforward neural network (FFNN), adapted for classification. The objective function is a cross-entropy-based function. Momentum is applied to the learning rate to speed up learning if changes in the same direction are made for many iterations. Finally, k-fold cross-validation is applied to get a sense of the model's performance.

Predictably, the less features (more pre-processing) are in the dataset, the better the classification accuracy. With two features, I was able to get comparable classification performance as the original authors (more or less 97% accuracy). With more features, it gets more interesting, since then the network needs to do more of its own pre-processing of the data. I also used a PCA transformation (commented out in the code) before training the network, with mixed results.

###TODO:

- Apply networks that take into account the temporal factor (either the same as the one used in the original paper, or different).
	+ in the original paper this gave improved performance
