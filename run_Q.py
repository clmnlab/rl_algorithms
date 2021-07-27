import gym
from models import QAgent
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = QAgent(eps = 0.01)
    n_episodes = 50000
    lose_draw_win = {-1:0, 0:0, 1:0}
    win_rates = []
    for i in range(n_episodes):
        if i>0 and i % 1000 ==0:
            pct = lose_draw_win[1]/i
            win_rates.append(pct)
            print(f'starting episode {i} win_rate {win_rates[-1]}')
        observation = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, action, reward))
            observation = observation_
        agent.update_Q()
        lose_draw_win[reward] += 1
    plt.plot(win_rates)
    plt.show()

