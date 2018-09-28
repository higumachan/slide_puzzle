import itertools
import numpy as np


class Evaluator:
    def __init__(self, w, h):
        pass
    
    def __call__(self, board):
        raise NotImplementedError


class ManhattanEvaluator(Evaluator):
    def __init__(self, w, h):
        self.board_pos = np.array([(x, y) for y, x in itertools.product(range(h), range(w))])
        self.correct_board_pos = self.board_pos.reshape(w, h, -1)
    
    def __call__(self, board):
        board = board.board

        distance = abs(self.board_pos[board] - self.correct_board_pos)

        return distance.sum()


if __name__ == '__main__':
    from board import Board, Direction
    board = Board([0, 1, 2 ,3, 4, 5, 6, 7, 8], 3, 3, 7, None)
    me = ManhattanEvaluator(3, 3)
    print(me(board))
    board.move(Direction.Up)
    print(me(board))
    board.move(Direction.Up)
    print(me(board))
    board.move(Direction.Up)
    print(me(board))

