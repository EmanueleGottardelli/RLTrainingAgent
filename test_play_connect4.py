import pygame
import random

from game import play
from players import RandomPlayer, MiniMaxPlayer, NNPlayer
from rl_agent import RLAgent
from connect4_state import Connect4State
from connect4_window import Connect4Window
from connect4_model import Connect4Model

# Setup RLAgent
agent = RLAgent(n_input=42, n_actions=7, epsilon=0.0)
agent.model.keras_model.load_weights("rl_connect4_model.keras")

# Avversari da testare
opponents = {
    "RandomPlayer": RandomPlayer(),
    "MiniMaxPlayer": MiniMaxPlayer(lookahead=1),
    "NNPlayer": NNPlayer(Connect4Model())
}

n_games = 50 # numero di partite

# Test play
for name, opponent in opponents.items():
    wins, losses, draws = 0,0,0
    for i in range(n_games):
        state = Connect4State()
        # alternanza chi inizia
        if i%2==0:
            states, _ = play(state, agent, opponent)
            winner = states[-1].state.winner()
        else:
            states, _ = play(state, opponent, agent)
            winner = states[-1].state.winner()*-1

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"\n VS {name}")
    print(f"Results: Wins: {wins}, Losses: {losses}, Draws: {draws}, WinRate: {wins/n_games:.2f}")

# Visualizzazione di una partita randomica contro avversario randomico
pygame.init()
window = Connect4Window(autoplayer=agent)
screen = pygame.display.set_mode((window.cols*window.grid_size, window.rows*window.grid_size))
pygame.display.set_caption("Play Connect4 Game")

# scelta avversario a caso
name, opponent = random.choice(list(opponents.items()))
print(f"\nPartita dimostrativa contro {name}")

state = Connect4State()
running = True

while running and not state.gameover():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if state.player() == 1:
        action = agent.next_action(state)
    else:
        action = opponent.next_action(state)

    state = state.move(action)
    window.state = state

    window.draw(screen)
    pygame.display.update()
    pygame.time.delay(500)

pygame.quit()

# Risultato finale
if state.winner() == 1:
    print("RLAgent ha vinto!")
elif state.winner() == -1:
    print("RLAgent ha perso!")
else:
    print("Pareggio!")
