import random
import chainer
from chainer.dataset import DatasetMixin
from board import Board, Direction, ZobristHash
from evaluator import ManhattanEvaluator
import numpy as np



class RandomMoveBoardManhattanDataset(DatasetMixin):
    def __init__(self, w, h, num=100000, reset_num=1000):
        evaluator = ManhattanEvaluator(w, h)
        
        visit_table = set()
        self.xs = []
        self.y = []
        zhash = ZobristHash(w, h)

        for i in range(int(num / reset_num)):
            self.board = Board([j for j in range(w * h)], w, h, 0, hasher=zhash)
            for j in range(reset_num):
                hs =  hash(self.board)
                k = 0
                while h in visit_table and k < 4:
                    self.move_random()
                    hs = hash(self.board)
                    k += 1
                visit_table.add(hs)
                self.xs.append(self.board.board.copy())
                self.y.append(evaluator(self.board))
                self.move_random()

    def move_random(self):
        while not self.board.move(random.choice([v for v in Direction])):
            pass

    def get_example(self, i):
        return self.xs[i], np.float32(self.y[i])

    def __len__(self):
        return len(self.xs)


if __name__ == '__main__':
    rmb = RandomMoveBoardManhattanDataset(5, 5, 10000)
    print(rmb.get_example(0))
    print(rmb.get_example(1))
    print(rmb.get_example(1000))
