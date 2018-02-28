from helper import ActorCritic
import keras.backend as K
import tensorflow as tf
import numpy as np
import test as t
import gym

# ============================== #
#        Hyperparameters         #
# ============================== #   
train_episodes = 1000
gamma = 0.9

epsilon = 1.0            # exploration probability at start
epsilon_decay = 0.95           

hidden_size = 256             # number of units in each Q-network hidden layer
learning_rate = 0.001         # Q-network learning rate

batch_size = 64                # experience mini-batch size
score_requirement = 0

# ============================== #
#           Training             #
# ============================== # 

print('Training starting..')
with tf.Session() as sess:
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make('BipedalWalker-v2')
    actor_critic = ActorCritic(env, sess,learning_rate, hidden_size, batch_size, epsilon, epsilon_decay, gamma)
    
    #actor_critic.load_actor()
    #actor_critic.load_critic()

    state = env.reset()
    action = env.action_space.sample()
    
    rewards_list=[]
    
    for episode in range(1, train_episodes):
        
        state = state[:14].reshape((-1, 14))
        action = actor_critic.act(state)
        new_state, reward, done, _ = env.step(action[0])
        new_state = new_state[:14].reshape((-1, 14))
        
        reward = np.float64(reward) #in case the enviroment gives us -100
        
        print('[Training]',
              'Episode: {}'.format(episode),
              'Total reward: {}'.format(reward))
        
        # Add experience to memory
        actor_critic.remember(state, action, reward, new_state, done)
        
        # ============================== #
        #          Finished Game         #
        # ============================== #    
        if done:   
            env.reset()
            action = env.action_space.sample()
            state, reward, done, _ = env.step(env.action_space.sample())
    
        # ======================================= #
        #          Reset in Stuck case            #
        # ======================================= #         
        if len(rewards_list) > 3:
            if ( round( reward, 4) == round( rewards_list[-1][1], 4) 
                 and round( rewards_list[-1][1], 4) == round( rewards_list[-2][1], 4) 
                 and round( rewards_list[-2][1], 4) == round( rewards_list[-3][1], 4) ):
                env.reset()
                action = env.action_space.sample()
                state, _, done, _ = env.step(env.action_space.sample())
                reward = -10.0
                
            
        actor_critic.train()
        
        rewards_list.append((episode, reward))
         
    # serialize weights to HDF5
    actor_critic.save_actor()
    actor_critic.save_critic()
                                 
    
# ============================================================ #
#          Visualizing the training progress                   #
# ============================================================ #   
#%matplotlib inline
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N


eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 1).T
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()



t.test(learning_rate, hidden_size, batch_size, epsilon, epsilon_decay, gamma)












