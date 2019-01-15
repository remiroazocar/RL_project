## Deep reinforcement learning final project (with DeepMind)

This project was developed using Python 3.5.0 and TensorFlow version 0.12. The OS utilised was Ubuntu Server 16.04. The following packages were utilised: matplotlib 1.5.3 (pyplot), scikit-learn 0.17.1, scikit-image 0.12.0 and numpy 1.11.0.

This directory contains a copy of the final report, model checkpoints (in the folder `models`, each in separate subdirectories for different questions) and the `code` folder, containing the programs that must be run for each question. To run the programs, first unzip the `models`, then execute the file corresponding to the relevant exercise. The training procedure has been commented out and the programs are ready to run a loaded model for each exercise. 

### Some notes:
* For Part A, the framework provided for model loading restores the model and plays 100 games (with maximum episode length of 150). The average episode length of the games is printed on screen when running the `code` files. Evidently, for most exercises, the episode length returned will be 150. For an example of an exercise that doesn’t, refer to `exerciseA8.py` (SARSA), which does not converge to the maximum number of episodes. 

* For A3, the final model restored is that corresponding to the best performing learning rate. 

* For A4, the final model restored corresponds to one individual run (in this case performed averaging over 100 games/trajectories)

* Model testing for A3-A8 consists of restoring the model and playing the game 100 times (with a maximum episode length of 150).

* For Part B, the framework provided for model loading restores a model for each game and results (numpy file containing in separate columns mean batch loss and in-game score) for each game. The Ms. Pacman model takes longer to restore because for the particular model loaded, metadata were stored in shorter intervals.
