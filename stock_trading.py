import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


def get_data():
    # returns a T x 3 list of stock prices
    # each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values


# create the experience replay buffer
class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, size):
        self.obs1_buffer = np.zeros([size, observation_dim], dtype=np.float32)  # stores the current state
        self.obs2_buffer = np.zeros([size, observation_dim], dtype=np.float32)  # stores the next state
        self.action_buffer = np.zeros(size, dtype=np.uint8)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.obs1_buffer[self.ptr] = state
        self.obs2_buffer[self.ptr] = next_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        indexes = np.random.randint(0, self.size, size=batch_size)  # create random indices of size 32
        return dict(s=self.obs2_buffer[indexes],
                    next_state=self.obs2_buffer[indexes],
                    action=self.action_buffer[indexes],
                    reward=self.reward_buffer[indexes],
                    done=self.done_buffer[indexes],
                    )


def get_scalar(env):
    states = []
    for _ in range(env.n_steps):
        action = np.random.choice(env.action_space)
        state, reward, done, _ = env.step(action)
        states.append(state)
        if done:
            break

        states.append(state)

    scalar = StandardScaler()
    scalar.fit(states)
    return scalar


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def multi_layer_perceptron(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
    i = Input(shape=(input_dim,))
    x = i

    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation="relu")(x)

    x = Dense(n_action)(x)

    model = Model(i, x)
    model.compile(loss='mse', optimizer='adam')
    return model


class MultiStockEnv:
    """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """

    def __init__(self, data, initial_investment=20000):
        self.stock_price_history = data
        self.n_steps, self.n_stocks = self.stock_price_history.shape

        self.investment = initial_investment
        self.cur_step = None
        self.cash_in_hand = None
        self.stocks_owned = None
        self.stock_price = None

        self.action_space = np.arange(3 ** self.n_stocks)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy

        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stocks)))
        self.state_dim = self.n_stocks * 2 + 1
        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_price = self.stock_price_history[self.cur_step]
        self.stocks_owned = np.zeros(self.n_stocks)
        self.cash_in_hand = self.investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        self.trade(action)

        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_steps - 1

        info = {'cur_value': cur_val}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stocks] = self.stocks_owned
        obs[self.n_stocks:2 * self.n_stocks] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.stocks_owned.dot(self.stock_price) + self.cash_in_hand

    def trade(self, action):
        action_vec = self.action_list[action]

        sell_idx = []
        buy_idx = []

        for i, a in enumerate(action_vec):
            if a == 0:
                sell_idx.append(i)
            elif a == 2:
                buy_idx.append(i)

        if sell_idx:
            for i in sell_idx:
                self.cash_in_hand += self.stock_price[i] * self.stocks_owned[i]
                self.stocks_owned[i] = 0
        if buy_idx:
            flag = True
            while flag:
                for i in buy_idx:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stocks_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        flag = False


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = multi_layer_perceptron(state_size, action_size)

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # get a random number, if its less than epsilon then choose a random action other get the model prediction
            return np.random.choice(self.action_size)
        act_vals = self.model.predict(state)
        return np.argmax(act_vals[0])

    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return

        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        next_states = minibatch['next_state']
        actions = minibatch['action']
        reward = minibatch['reward']
        done = minibatch['done']
        target = reward + (1-done) *self.gamma * np.argmax(self.model.predict(next_states), axis=1)


        target_in_full = self.model.predict(states)  # getting all targets

        target_in_full[
            np.arange(batch_size), actions] = target  # setting the targets for the batch we just went through

        self.model.train_on_batch(states, target_in_full)  # training

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)  # add to the buffer
            agent.replay(batch_size)  # one step of gradient descent
        state = next_state

    return info['cur_val']


if __name__ == '__main__':
    # conifg
    models_folder = 'rl_trader_models'
    rewards_folder = 'rl_trader_rewards'
    initial_investment = 20000
    batch_size = 32
    n_eps = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test')

    args = parser.parse_args()
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scalar(env)

    portfolio_value = []

    if args.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        env = MultiStockEnv(test_data, initial_investment)

        agent.epsilon = 0.01
        agent.load(f'{models_folder}/dqn.h5')

    for e in range(n_eps):
        t0 = datetime.now()
        print("I am in train")
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f'episode: {e + 1}/{n_eps}, episode end value: {val: .2f}, duration: {dt}')
        portfolio_value.append(val)
    if args.mode == 'train':
        agent.save(f'{models_folder}/dqn.h5')

        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f'rl_trader_rewards/{args.mode}.npy', portfolio_value)
