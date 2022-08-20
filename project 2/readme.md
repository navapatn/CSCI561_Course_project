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

After loading raw data, I performed data normalization so that all features have equal weight. Otherwise feature, for example, Income would be considered much more important compared to Education.

## Segmentation via Clustering using Customer Data
This section is to perform segmentation to see how our customers can be grouped.
After normalizing the data, I performed clustering as a preliminary analysis of the data to see how the groups would look like from these algorithms

I tried tree different models:
*	**Hierarchical Clustering** – to get an idea of the ideal number of clusters
*	**K-mean Clustering** – after getting a decent idea of the expected clusters, perform K-mean to see how the groups result look like.
*	**K-mean Clustering with PCA** – to reduce number of features to only the most important ones.

![Hierarchicalresult](/images/ca-Hier.JPG)
![KmeanWCSS](/images/ca-kmeanwcss.JPG)
![Kmeanclusters](/images/ca-kmean.JPG)


## Technique 2: Q-learning
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![EDA1](/images/ca-EDA1.JPG)
![EDA2](/images/ca-EDA2.JPG)
![EDA3](/images/ca-EDA3.JPG)
![EDA4](/images/ca-EDA4.JPG)
![EDA5](/images/ca-EDA5.JPG)


## Results

*	**Logistic Regression** - is used to calculate purchase probability relative to price elasticity.

Price Elasticity = % change in purchase probability / % change in price.

Inelastic, we increase the price of the product since it affects the probability of buying less than 1% per 1% price change.

Elastic, we decrease the price of the product since it affects the probability of buying less than 1% per 1% price change.

![purchaseprob](/images/ca-purchaseprob.JPG)

This way we know which products we should increase/decrease in price or add promotions to them

This can be done by customer segments/brands/etc.
