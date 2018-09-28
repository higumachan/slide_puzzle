import copy
from board import Board, Direction, ZoblistHash
from evaluator import ManhattanEvaluator

class BeamSearch:
    def __init__(self, board, evaluator, beam_width):
        self.board = board
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.visit_table = set()

    def search(self):
        states = [(self.evaluator(self.board), self.board, [])]
        while not self.is_complete(states):
            nexts = self.next_states(states)
            nexts.sort(key=lambda x: x[0])
            states = nexts[:self.beam_width]
            print(states[0][0])
        return list(filter(lambda x: x[1].completed(), states))[0]

    def next_states(self, states):
        results = []
        for state in states:
            for direction in Direction:
                board = copy.deepcopy(state[1])
                if board.move(direction):
                    hs = hash(board)
                    if hs not in self.visit_table:
                        results.append((self.evaluator(board), board, state[2] + [direction]))
                        self.visit_table.add(hs)
        return results

    def is_complete(self, states):
        return any([state[1].completed() for state in states])


if __name__ == '__main__':
    w, h = 5, 5
    z = ZoblistHash(w, h)
    me = ManhattanEvaluator(w, h)
    board = Board([i for i in range(w * h)], w, h, 4, z)
    board.move(Direction.Up)
    board.move(Direction.Up)
    board.move(Direction.Left)
    board.move(Direction.Down)
    bs = BeamSearch(board, me, 100)
    print(bs.search())

    from dataset import RandomMoveBoardManhattanDataset

    dataset = RandomMoveBoardManhattanDataset(w, h, 1000, 1000)
    b, d = dataset.get_example(999)
    dataset.board.print_board()
    bs = BeamSearch(dataset.board, me, 1000)
    result = bs.search()
    print(result, len(result[2]))
