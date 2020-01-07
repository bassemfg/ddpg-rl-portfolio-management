#!/usr/bin/env python
# coding: utf-8

# # Deep Deterministic Policy Gradient for Portfolio Management

# **Steps to try:**
# 1. We first try to overfit 16 stocks using 3 years of training data.
# 2. Try to generalize to other years of the same stock.
# 3. Get some insigts on network topology and hyperparameter tuning.
# 
# **Possible improvement methods:**
# 1. Use correlated action noise
# 2. Use adaptive parameter noise
# 
# **Figures to show:**
# 1. Training: total rewards w.r.t episode
# 2. How the model performs on training data
# 3. How the model performs on testing data

# ## Setup

# In[1]:


import os
#os.chdir('/mnt/c/c/Users/byacoube/source/repos/capstone/src')
os.chdir(r'C:\Users\byacoube\source\repos\capstone\src')
#from google.colab import drive
#drive.mount('/content/drive')
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib.pyplot as plt
import matplotlib
# for compatible with python 3
#from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Seaborn, useful for graphics
import seaborn as sns

# Import Bokeh modules for interactive plotting
import bokeh.io
#import bokeh.mpl
import bokeh.plotting

# Magic function to make matplotlib inline; other style specs must come AFTER
get_ipython().run_line_magic('matplotlib', 'inline')

# This enables SVG graphics inline.  There is a bug, so uncomment if it works.
# %config InlineBackend.figure_formats = {'svg',}

# This enables high resolution PNGs. SVG is preferred, but has problems
# rendering vertical and horizontal lines
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")
matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend', fontsize=20)
# configure Seaborn settings 
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'EFEFD5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)

# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# In[3]:


# read the data and choose the target stocks for training a toy example
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]
num_training_time = history.shape[1]
num_testing_time = history.shape[1]
window_length = 3


#print(history.shape,num_testing_time,num_training_time)


# In[5]:


# dataset for 16 stocks by splitting timestamp
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]

# 16 stocks are all involved. We choose first 3 years as training data
num_training_time = 1095
target_stocks = abbreviation
target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))

for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

# and last 2 years as testing data.
testing_stocks = abbreviation
testing_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time, 
                               history.shape[2]))
for i, stock in enumerate(testing_stocks):
    testing_history[i] = history[abbreviation.index(stock), num_training_time:, :]
"""
print(target_history.shape)
print(target_history)
print(testing_history.shape)
print(testing_history)
"""


# In[6]:


nb_classes = len(target_stocks) + 1


# In[7]:


# visualize stock prices
if True:
    date_list = [index_to_date(i) for i in range(target_history.shape[1])]
    x = range(target_history.shape[1])
    for i in range(len(target_stocks)):
        plt.figure(i)
        plt.plot(x, target_history[i, :, 1])  # open, high, low, close = [0, 1, 2, 3]
        plt.xticks(x[::200], date_list[::200], rotation=30)
        plt.title(target_stocks[i])
        plt.show()


# ## Load Models

# In[8]:


import tensorflow as tf
tf.__version__


# In[9]:


from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn
import tensorflow as tf

from stock_trading import StockActor, StockCritic, obs_normalizer, get_model_path, get_result_path,test_model, get_variable_scope, test_model_multiple
    
from model.supervised.lstm import StockLSTM
from model.supervised.cnn import StockCNN


# In[10]:


# common settings
batch_size = 64
action_bound = 1.
tau = 1e-3


# In[11]:


models = []
model_names = []
window_length_lst = [3]
predictor_type_lst = ['lstm']
use_batch_norm = True


# In[13]:


# instantiate environment, 3 stocks, with trading cost, window_length 3, start_date sample each time
for window_length in window_length_lst:
    for predictor_type in predictor_type_lst:
        name = 'DDPG_window_{}_predictor_{}'.format(window_length, predictor_type)
        model_names.append(name)
        tf.reset_default_graph()
        sess = tf.Session()
        tflearn.config.init_training_mode()
        action_dim = [nb_classes]
        state_dim = [nb_classes, window_length]
        variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)
        with tf.variable_scope(variable_scope):
            actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size, predictor_type, 
                               use_batch_norm)
            critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                                 learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(), 
                                 predictor_type=predictor_type, use_batch_norm=use_batch_norm)
            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

            model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
            summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

            ddpg_model = DDPG(None, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                              config_file='config/stock.json', model_save_path=model_save_path,
                              summary_path=summary_path)
            ddpg_model.initialize(load_weights=True, verbose=True)
            models.append(ddpg_model)


# #### Note that the following tests use the dataset from the following cell. To run the original dataset, you have to change the date or simulation steps in order to avoid exceptions.

# In[14]:


# create another dataset
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target_2.h5')
history = history[:, :, :4]
nb_classes = len(history) + 1
print(history.shape)
testing_history = history
testing_stocks = abbreviation
target_history = history
target_stocks = abbreviation
from environment.portfolio import PortfolioEnv, MultiActionPortfolioEnv


# In[15]:



env = MultiActionPortfolioEnv(target_history, target_stocks, model_names[:1], steps=1500, 
                              sample_start_date='2012-10-30')


# In[16]:


test_model_multiple(env, models[:1])


# In[33]:


# evaluate the model with unseen data from same stock, fixed the starting date
env = MultiActionPortfolioEnv(testing_history, testing_stocks, model_names[:1], steps=650, 
                              start_idx=num_training_time, sample_start_date=None)


# In[34]:


test_model_multiple(env, models[:1])



# In[37]:


from environment.portfolio import MultiActionPortfolioEnv
selected_models = [models[0] ]
selected_model_names = [model_names[0]]
env = MultiActionPortfolioEnv(target_history, target_stocks, selected_model_names, steps=1500, 
                              start_idx=0, sample_start_date='2012-10-30')
test_model_multiple(env, selected_models)


# In[38]:


import model.supervised.imitation_optimal_action as imitation_optimal_action





