# Go-5x5 AI agents: Project Overview
* Created AI agents based on Search, Game Playing, and Reinforcement Learning to compete in an in-class online tournaments. The algorithms chosen for this project include

1. Alpha-Beta Pruning
2. Q-learning

* Engineered features from the official Go game rule and various techniques such as Liberty, Komi, Passing, etc. to help AI agents decide what is the best move to play next

## Code and Resources Used 
**Python Version:** 3.7  

## Basic Function

To evaluate game states in each turn, we need multiple functions to do that.

 **Game state evaluation function**
*	detect_neighbor
*	find_ally
*	find_liberty
*	remove_died_piece
*	next_state_after_moved
*	valid_place_check
*	get_legal_state

these functions act as a data pulling function to give the AI agents a set of data in each turn

## Technique 1: Alpha-Beta Pruning

This method is used to select the best path for us and the worst path for the opponent. Heuristic function is being used to calculate points of each turn.

The Heuristic function is defined as (#number of our color - #number of opponent color after picking move x).

![alphabeta](/images/abpruning.png)
![alphabeta](/images/alphabeta.JPG)

The algorithm will recursively calculate point after making all possible moves and select best outcome for us and worst outcome for the opponent.

## Technique 2: Q-learning

This method is also tested to select best move at certain states by learning from multiple board scenarios.

Q-valued is re-calculated every time the game ends, if the agent lose, it will backpropagate negative value to all previous states.
However, if the agent win the game, it will backpropagate positive value to all previous states. This method requires multiple rounds of training to make the agent able to learn all possible states from the board.

![alphabeta](/images/qlearning.png)


## Results

* The agent competed against **six different types of agent**
*	random
*	greedy
*	aggressive
*	Alpha-Beta
*	Q-learning
*	Championship

The results were calculated on multiple runs in a simulation program, and the agent achieved about 90% win-rate total
