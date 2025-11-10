import numpy as np
import players as Player
from model import Model

class RLAgent(Player.Player):
    def __init__(self, n_input, n_actions, epsilon=0.2, gamma=0.99):
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.model = Model(ninput=n_input, layers=[64,128,128,128,128])
        self.replay_buffer = []

    def next_action(self, state):
        actions = state.actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            states_list = [state.move(a).cells for a in actions]
            q_values = self.model.predict(states_list)
            current_player = 1 if state.player() == 1 else 2
            player_q = [q[current_player] for q in q_values]
            return actions[np.argmax(player_q)]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state.cells, action, reward,
                                   next_state.cells if next_state else None, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        batch_idx = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        states = []
        targets = []

        for idx in batch_idx:
            state, action, reward, next_state, done = self.replay_buffer[idx]
            state = np.array(state)
            q_pred = self.model.predict([state])[0]
            if done or next_state is None:
                q_pred[action] = reward
            else:
                next_state = np.array(next_state)
                q_next = self.model.predict([next_state])[0]
                q_pred[action] = reward + self.gamma * np.max(q_next)
            states.append(state)
            targets.append(q_pred)

        self.model.keras_model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
