import gym
import numpy as np
from utils import plot_learning_curves
from models import Agent
if __name__ =='__main__':
    env = gym.make('CartPole-v1')
    n_games = 5000
    scores = []
    eps_history = []
    agent = Agent(input_dims = env.observation_space.shape,
                  n_actions = env.action_space.n, lr =1e-4)
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i}: score {score}, avg_score {avg_score}')
    x = [k + 1 for k in range(len(scores))]
    plot_learning_curves(scores, x, 'test_cardpole')
