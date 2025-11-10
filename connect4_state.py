from state import make_rules, State

connect4_rules = make_rules(cols=7, rows=6, score=4)


class Connect4State(State):

    def __init__(self, cells=None):
        super().__init__(cells, cols=7, rows=6, rules=connect4_rules)

    def actions(self):
        return [i for i in range(self._cols) if 0 in self.col(i)]

    def col(self, i):
        return self.cells[i::self._cols]

    def move(self, action):
        state = type(self)(self.cells)
        row = self._rows - 1 - list(reversed(state.col(action))).index(0)
        return super().move(row*7 + action)


if __name__ == '__main__':
    from players import ConsolePlayer, MiniMaxPlayer
    from game import play
    states, _ = play(Connect4State(), ConsolePlayer(), MiniMaxPlayer(4))
    print(states[-1].state)