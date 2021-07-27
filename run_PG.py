import gym
import matplotlib.pyplot as plt
import numpy as np
from models import PolicyGradientAgent





if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[8], n_actions=4)
    fname = 'lunar_test' + str(agent.lr) + '_' + str(n_games)
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
            agent.store_rewards(reward)
            observation = observation_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f'episode {i} score {score}, average score {avg_score}')
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curves(scores, x, figure_file)
