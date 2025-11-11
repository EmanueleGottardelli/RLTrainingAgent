from model import Model


class Connect4Model(Model):
    def __init__(self, model=None):
        super().__init__(ninput=42,
                         layers=[64, 128, 128, 128, 128],
                         n_actions=7,
                         model=model)