import random
import chainer
from chainer.dataset import DatasetMixin
from board import Board, Direction
from evaluator import ManhattanEvaluator
import numpy as np



class RandomMoveBoardManhattanDataset(DatasetMixin):
    def __init__(self, w, h, num=100000):
        self.board = Board([i for i in range(w * h)], w, h, 0)
        evaluator = ManhattanEvaluator(w, h)
        
        visit_table = set()
        self.xs = []
        self.y = []

        for i in range(num):
            h =  hash(self.board)
            while h in visit_table:
                self.move_random()
                h =  hash(self.board)
            visit_table.add(h)
            self.xs.append(self.board.board.copy())
            self.y.append(evaluator(self.board))
            self.move_random()

    def move_random(self):
        while True:
            try:
                self.board.move(random.choice([v for v in Direction]))
            except:
                continue
            break

    def get_example(self, i):
        return self.xs[i], np.float32(self.y[i])

    def __len__(self):
        return len(self.xs)


if __name__ == '__main__':
    rmb = RandomMoveBoardManhattanDataset(5, 5, 10000)
    print(rmb.get_example(0))
    print(rmb.get_example(1))
    print(rmb.get_example(1000))
