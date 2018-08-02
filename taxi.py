import numpy as np
import gym
import random
import time

env = gym.make("Taxi-v2")
env.render()

action_space = env.action_space.n
print(action_space)
state_space = env.observation_space.n
print(state_space)

q_table = np.zeros(action_space*state_space).reshape(state_space, action_space)

max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

def traning(max_episodes, max_steps, learning_rate, gamma, epsilon):
    show_episodes = [499, 49999]
    for i_episode in range(max_episodes):
        sum_reward = 0
        cur_state = env.reset()
        done = False
        if i_episode in show_episodes:
            print('Start episode {}:'.format(i_episode))
            env.render()
            # print('state={}'.format(cur_state))
        
        for i_step in range(max_steps):
            if (random.random() < epsilon):
#               exploration 
                action = env.action_space.sample()
            else:
#               exploitation
                action = np.argmax(q_table[cur_state, :])
#           simulate action, observe new_state, reward,..
            new_state, reward, done, info = env.step(action)
            if i_episode in show_episodes:
                time.sleep(1.5)
                print('Step {}:'.format(i_step))
                env.render()
                print('Total reward = {}'.format(sum_reward+reward))
#           update q-table     
            q_table[cur_state, action] = (1 - learning_rate) * q_table[cur_state, action] + \
                learning_rate * (reward + gamma * np.max(q_table[new_state]))
#           update state
            cur_state = new_state
            sum_reward += reward
            
            if done or i_step == max_steps-1:
                if i_episode in show_episodes:
                    print('End episode {}! with sum_reward={}, sum_step={}'.format(i_episode, sum_reward, i_step))
                break
#           update epsilon(k)        
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*(i_episode+1))


def playing(episode_num):
    
    for i_episode in range(episode_num):
        sum_reward = 0
        done = False
        cur_state = env.reset()
        print('Start episode {}:'.format(i_episode))
        env.render()
        
        for i_step in range(30):
            
            time.sleep(1)
            print('step {}:'.format(i_step))
            action = np.argmax(q_table[cur_state])
            new_state, reward, done, info = env.step(action)
            env.render()
            cur_state = new_state
            sum_reward += reward
            if (done):
                print('End episode {}! with sum_reward={}, sum_step={}'.format(i_episode, sum_reward, i_step))
                break

traning(50000, 100, 0.7, 0.681, 1.0)
# playing(1)