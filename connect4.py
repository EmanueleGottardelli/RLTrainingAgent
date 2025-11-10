import click
from players import ConsolePlayer, MiniMaxPlayer, NNPlayer
from game import play, simulate
from connect4_state import Connect4State


@click.command('connect4')
@click.option('--simulations', '-s',
              default=10000,
              help='How many plays to simulate for training.')
@click.option('--mode', '-m',
              default='window',
              type=click.Choice(['console', 'window']),
              help='Starts game in a terminal or a window.')
@click.option('--ai-player', '-p',
              default='nn',
              type=click.Choice(['nn', 'minimax']),
              help='Number of epochs to train.')
@click.option('--epochs', '-e',
              default=3,
              help='Number of epochs to train. (only for ai-player=nn)')
@click.option('--lookahead', '-l',
              default=3,
              help='Lookahead depth for the minimax algorithm.'
              ' (only for ai-player=minimax)')
def connect4(simulations, mode, ai_player, epochs, lookahead):
    state = Connect4State()

    if ai_player == 'nn':
        from connect4_model import Connect4Model
        model = Connect4Model()
        plays = simulate(state, simulations)
        model.train(plays, epochs=epochs)
        autoplayer = NNPlayer(model)
    else:
        autoplayer = MiniMaxPlayer(lookahead=lookahead)

    if mode == 'console':
        states, _ = play(state, ConsolePlayer(), autoplayer)
        print(states[-1].state)
    else:
        from connect4_window import Connect4Window
        Connect4Window(autoplayer=autoplayer).show()


if __name__ == '__main__':
    connect4()