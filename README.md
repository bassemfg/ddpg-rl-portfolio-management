


# Using Reinforcement Learning for Portfolio Optimization

# 

## Algorithms used in this work
* Long short term memory
* Deep Deterministic Policy Gradient

## Dataset
* S&P500 dataset from kaggle found [here](https://www.kaggle.com/camnugent/sandp500)

## References
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
* The code is inspired by [CSCI 599 deep learning and its applications final project
](https://github.com/vermouth1992/drl-portfolio-management) 
* The environment is inspired by [wassname/rl-portfolio-management](https://github.com/wassname/rl-portfolio-management)
* DDPG implementation is inspired by [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To install the required python packges, browse to the code folder then run ```pip install --user --requirement requirements.txt```

### Running the tests

ddpg_tests.ipynb is a step by step jupyter notebook showing the performance of the trained agent on unseen stocks. You can run this jupyter notebook directly without having to run the training since the training weights are saved.


### Running the training 

To train the model from scratch and overwrite the saved weights, run stock_trading.py. This could take several hours.

## License

This project is licensed under the MIT License.
