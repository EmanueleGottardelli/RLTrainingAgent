import numpy as np
from players import Player
from model import Model


class RLAgent(Player):
    def __init__(self, n_input, n_actions, epsilon=0.2, gamma=0.99):
        self.epsilon = epsilon  # probabilità di esplorazione
        self.gamma = gamma  # fattore di sconto
        self.n_actions = n_actions
        self.model = Model(ninput=n_input, layers=[64, 128, 128, 128, 128],n_actions=n_actions)
        self.replay_buffer = []

    def next_action(self, state):
        actions = state.actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            # Convertiamo la lista di stati in array NumPy
            states_list = [state.move(a).cells for a in actions]
            states_array = np.array(states_list, dtype=np.float32)
            q_values = self.model.predict(states_array)

            # Player index: 1 -> indice 1, -1 -> indice 2 (per la tua codifica softmax)
            current_player = 1 if state.player() == 1 else 2
            player_q = [q[current_player] for q in q_values]

            return actions[np.argmax(player_q)]

    def store_transition(self, state, action, reward, next_state, done):
        # Convertiamo gli stati in array NumPy prima di memorizzare
        state_arr = np.array(state.cells, dtype=np.float32)
        next_state_arr = np.array(next_state.cells, dtype=np.float32) if next_state else None

        self.replay_buffer.append((state_arr, action, reward, next_state_arr, done))
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
            state = np.array(state, dtype=np.float32)
            q_pred = self.model.predict(np.array([state], dtype=np.float32))[0]

            if done or next_state is None:
                q_pred[action] = reward
            else:
                next_state = np.array(next_state, dtype=np.float32)
                q_next = self.model.predict(np.array([next_state], dtype=np.float32))[0]
                q_pred[action] = reward + self.gamma * np.max(q_next)

            states.append(state)
            targets.append(q_pred)

        self.model.keras_model.fit(
            np.array(states, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            epochs=1,
            verbose=0
        )
