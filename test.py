from helper import ActorCritic
import tensorflow as tf
import keras.backend as K
import gym

def test(learning_rate = 0.001, hidden_size = 24,batch_size = 32, epsilon = 1.0, 
                 epsilon_decay = .995, gamma = .95):   
    
    env = gym.make('BipedalWalker-v2')
    # Testing
    test_episodes = 10
    test_max_steps = 200
    env.reset()
    with tf.Session() as sess:
                
        K.set_session(sess)
        actor_critic = ActorCritic(env, sess,learning_rate, hidden_size, batch_size, epsilon, epsilon_decay, gamma)
        actor_critic.load_actor()
        actor_critic.load_critic()
        
        #take 1st random move
        state, _, done, _ = env.step(env.action_space.sample())
        for _ in range(1, test_episodes):
            current_step = 0
            while current_step < test_max_steps:
                env.render()
                
                state = state[:14].reshape((-1, 14))
                
                action = actor_critic.actor_model.predict(state)[0]
                
                next_state, _, done, _ = env.step(action)
                if done:
                    current_step = test_max_steps
                    env.reset()
                    state, _, done, _ = env.step(env.action_space.sample())
    
                else:
                    state = next_state
                    current_step += 1

#test()                
                    

