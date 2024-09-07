import chess
import chess.svg
import chess.engine
from Engines.RollOutV1 import eval

class Agent():
    def __init__(self):
        self.dummy=0

    def evaluate_board(self,board):
        if not board.is_checkmate():
            return eval.eval(board)
        else:
            if not board.is_stalemate():
                if board.turn:
                    return -10000
                else:
                    return 10000

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

    def find_best_move(self,board):
        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = float('-inf')
        ply_depth = 4
        for move in legal_moves:
            board.push(move)
            eval = self.minimax(board, ply_depth - 1, False)
            board.pop()

            if eval > best_eval:
                best_eval = eval
                best_move = move

        return best_move,eval

if __name__=='__main__':
    custom_fen = 'rnbqkbnr/1ppppppp/p7/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2'
    board = chess.Board(fen=custom_fen)

    bot = Agent()
    eva = bot.evaluate_board(board)
    print(eva)
    print(bot.find_best_move(board))