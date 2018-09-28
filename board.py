import numpy as np
from enum import Enum
import functools

class Direction(Enum):
    Up = (-1, 0)
    Left = (0, -1)
    Down = (1, 0)
    Right = (0, 1)


class ZoblistHash(object):
    def __init__(self, w, h):
        self.table = np.random.randint(0, 1 << 63, (w * h, w * h, w * h), dtype=np.uint64)

    def __call__(self, board):
        return functools.reduce(np.bitwise_xor, self.table[board.board.reshape(-1), np.arange(board.w * board.h), board.target_number])


class Board(object):
    def __init__(self, numbers, w, h, target_number, hasher):
        self.w = w
        self.h = h
        self.board = np.array(numbers).reshape((w, h))
        index = (self.board == target_number).argmax()
        self.target = np.array((index % w, index // w))
        self.hasher = hasher
        self.target_number = target_number

    def move(self, direction):
        a = self.target + np.array(direction.value)
        self.board[a[0], a[1]], self.board[self.target[0], self.target[1]] = self.board[self.target[0], self.target[1]], self.board[a[0], a[1]]
        self.target = a

    def __hash__(self):
        return self.hasher(self)

    def print_board(self):
        print(self.board)


if __name__ == '__main__':
    np.random.seed(100)
    z = ZoblistHash(3, 3)
    board = Board([0, 1, 2 ,3, 4, 5, 6, 7, 8], 3, 3, 4, z)
    board.print_board()
    print(hash(board))
    board.move(Direction.Up)
    board.print_board()
    print(hash(board))
    board = Board([0, 1, 2 ,3, 4, 5, 6, 7, 8], 3, 3, 6, z)
    board.print_board()
    print(hash(board))
