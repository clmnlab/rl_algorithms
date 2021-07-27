import gym
import matplotlib.pyplot as plt
import numpy as np
from models import ActorCriticAgent
from utils import plot_learning_curves

if __name__ == '__main__':
    n_games = 3000
    env = gym.make('LunarLander-v2')
    agent = ActorCriticAgent(lr=5e-6, input_dims=[8], n_actions=4, fc1_dims=2048, fc2_dims=1024)
    fname = 'Actor_Critic'+ 'lunar_test' + str(agent.lr) + '_' + str(n_games)
    figure_file = fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f'episode {i} score {score}, average score {avg_score}')
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curves(scores, x, figure_file)
