import chess.polyglot
import chess

class TranspositionTable:
    def __init__(self):
        self.table = {}

    def lookup(self, board):
        """Lookup the board position in the transposition table."""
        hash_value = chess.polyglot.zobrist_hash(board)
        return self.table.get(hash_value, None)

    def store(self, board, prob, score, flag, best_move,principle_variation):
        """Store the evaluated board position in the transposition table."""
        hash_value = chess.polyglot.zobrist_hash(board)
        self.table[hash_value] = (prob, score, flag, best_move,principle_variation)

    def clear(self):
        """Clear the transposition table."""
        self.table.clear()


if __name__ == '__main__':
    table = TranspositionTable()
    board = chess.Board()
    if table.lookup(board):
        print(True)
    else:
        print(False)