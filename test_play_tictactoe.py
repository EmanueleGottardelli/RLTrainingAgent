import pygame
import random
from game import play
from players import RandomPlayer, MiniMaxPlayer, NNPlayer
from rl_agent import RLAgent
from tictactoe_state import TicTacToeState
from tictactoe_window import TicTacToeWindow
from tictactoe_model import TicTacToeModel

# Setup RLAgent
agent = RLAgent(n_input=9, n_actions=9, epsilon=0.0)
agent.model.keras_model.load_weights("rl_tictactoe_model.keras")

# Avversari da testare
opponents = {
    "RandomPlayer": RandomPlayer(),
    "MiniMaxPlayer": MiniMaxPlayer(lookahead=3),
    "NNPlayer": NNPlayer(TicTacToeModel())
}

n_games = 50  # numero di partite

# Test play
for name, opponent in opponents.items():
    wins, losses, draws = 0, 0, 0
    for i in range(n_games):
        state = TicTacToeState()
        # alternanza chi inizia
        if i % 2 == 0:
            states, _ = play(state, agent, opponent)
            winner = states[-1].state.winner()
        else:
            states, _ = play(state, opponent, agent)
            winner = states[-1].state.winner() * -1  # punto di vista RLAgent

        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"\n VS {name}")
    print(f"Results: Wins: {wins}, Losses: {losses}, Draws: {draws}, WinRate: {wins/n_games:.2f}")

# Visualizzazione grafica di una partita randomica contro avversario randomico
pygame.init()
window = TicTacToeWindow(autoplayer=agent)
screen = pygame.display.set_mode((window.cols*window.grid_size, window.rows*window.grid_size))
pygame.display.set_caption("Demo RLAgent TicTacToe")

# scelta avversario a caso
name, opponent = random.choice(list(opponents.items()))
print(f"\nPartita dimostrativa contro {name}")

state = TicTacToeState()
running = True

while running and not state.gameover():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Mossa del giocatore corrente
    if state.player() == 1:
        action = agent.next_action(state)
    else:
        action = opponent.next_action(state)

    state = state.move(action)
    window.state = state

    # Disegna tutto
    window.draw(screen)
    pygame.display.update()
    pygame.time.delay(500)  # ritardo per vedere le mosse

pygame.quit()

# Risultato finale
if state.winner() == 1:
    print("RLAgent ha vinto!")
elif state.winner() == -1:
    print("RLAgent ha perso!")
else:
    print("Pareggio!")
