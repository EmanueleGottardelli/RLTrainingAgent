
from pygame.draw import circle
from connect4_state import Connect4State
from players import MiniMaxPlayer
from board_window import BoardWindow

CELL_COLORS = [(255, 255, 255),  # Empty cell
               (180, 40, 30),    # Player 1
               (180, 180, 30)]   # Player -1
BOARD_COLOR = (30, 50, 140)


class Connect4Window(BoardWindow):

    def __init__(self, state=None, autoplayer=None):
        super().__init__(
            title='Connect 4',
            state=state or Connect4State(),
            autoplayer=autoplayer or MiniMaxPlayer(3),
            cols=7, rows=6,
            grid_size=70, cell_padding=3,
            padding_v=10, padding_h=10)

    def action_for_cell(self, pos):
        return pos.col

    def draw_background(self, screen):
        screen.fill(BOARD_COLOR)

    def draw_cell(self, screen, player, rect):
        radius = int(rect.width/2)
        center = self.center_for_rect(rect)
        circle(screen, CELL_COLORS[player], center, radius)


if __name__ == '__main__':
    window = Connect4Window()
    window.show()
    print('hello')