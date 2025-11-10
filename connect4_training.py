from rl_agent import RLAgent
from connect4_state import Connect4State
from players import RandomPlayer
from game import play

agent = RLAgent(n_input=42, n_actions=7, epsilon=0.2, gamma=0.99)
opponent = RandomPlayer()

n_episodes = 5000

for episode in range(n_episodes):
    state = Connect4State()
    done = False

    while not state.gameover():
        if state.player == 1:
            action = agent.next_action(state)
            next_state = state.move(action)
            reward = 0
            done = next_state.gameover()
            if done:
                winner = state.winner()
                reward = 1 if winner == 1 else -1
            agent.store_transition(state, action, reward, next_state, done)
        else:
            action = opponent.next_action(state)
            next_state = state.move(action)

        state=next_state

    agent.train(batch_size=64)

    if(episode+1)%100==0:
        print(f"Episode: {episode+1} completed")

agent.model.keras_model.save("rl_connect4_model.h5")
print("Training Connect4 completato e modello salvato")