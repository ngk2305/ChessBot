import chess
import chess.svg
import chess.engine

class Agent():
    def __init__(self,depth):
        self.depth= depth

    def evaluate_board(self,board):
        evaluation = 0
        if not board.is_checkmate():
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9,"K":20}.get(piece.symbol().upper(), 0)
                    evaluation += value if piece.color else -value

        else:
            if not board.is_stalemate():
                if board.turn:
                    evaluation=100
                else:
                    evaluation=-100
        return evaluation

    def minimax(self,board, depth, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval)
            return min_eval

    def find_best_move(self,board, depth):
        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = float('-inf')

        for move in legal_moves:
            board.push(move)
            eval = self.minimax(board, depth - 1, False)
            board.pop()

            if eval > best_eval:
                best_eval = eval
                best_move = move

        return best_move