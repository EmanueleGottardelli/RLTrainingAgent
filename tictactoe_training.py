# train_rl_tictactoe.py
from rl_agent import RLAgent
from tictactoe_state import TicTacToeState
from players import RandomPlayer
from game import play

# Parametri agente RL
agent = RLAgent(n_input=9, n_actions=9, epsilon=0.2, gamma=0.99)
opponent = RandomPlayer()

n_episodes = 5000

for episode in range(n_episodes):
    state = TicTacToeState()
    done = False

    while not state.gameover():
        if state.player() == 1:
            action = agent.next_action(state)
            next_state = state.move(action)
            reward = 0
            done = next_state.gameover()
            if done:
                winner = next_state.winner()
                reward = 1 if winner == 1 else -1
            agent.store_transition(state, action, reward, next_state, done)
        else:
            action = opponent.next_action(state)
            next_state = state.move(action)

        state = next_state

    agent.train(batch_size=64)

    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1} completed")

# Salva il modello allenato
agent.model.keras_model.save("rl_tictactoe_model.h5")
print("Training TicTacToe completato e modello salvato.")
