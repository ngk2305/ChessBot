import chess
import chess.svg
import chess.engine
from Engines.NeuralEngineV4 import getBoard
import torch
from Engines.NeuralEngineV4 import NEv4
from Engines.NeuralEngineV4 import Search
class Agent():
    def __init__(self):
        self.NN= NEv4.ChessEvaluator()
        self.NN.load_state_dict(torch.load(f'Engines/NeuralEngineV4/NEv4_weights.pth'))


    def evaluate_board(self,board):
        evaluation = 0
        if not board.is_checkmate():
            evaluation = self.NN(torch.Tensor(getBoard.get_bit_board(board)),torch.Tensor(getBoard.get_info_board(board)))

        else:
            if not board.is_stalemate():
                if board.turn:
                    evaluation=0
                else:
                    evaluation=1
        return evaluation

    def minimax_alpha_beta(self, board, probability, alpha, beta, maximizing_player):

        if probability <= 0.00002 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = Search.simpleMoveOrder(board)
        if maximizing_player:
            max_eval = 0
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, probability/len(legal_moves), alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Prune remaining branches
            return max_eval
        else:
            min_eval = 1
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, probability/len(legal_moves), alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Prune remaining branches
            return min_eval

    def find_best_move(self, board):
        legal_moves = Search.simpleMoveOrder(board)

        best_move = None
        if board.turn == chess.WHITE:  # Maximizing player (white)
            max_eval = 0
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, 1, 0, 1, False)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    if max_eval==1:
                        break
        else:  # Minimizing player (black)
            min_eval = 1
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, 1, 0,1, True)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    if min_eval==0:
                        break

        return best_move,eval






if __name__== '__main__':
    board = chess.Board(fen='rnbqkbnr/ppppp1pp/5p2/4P3/8/3B4/PPPP1PPP/RNBQK1NR w KQkq - 1 4')

    new_bot = Agent()
    new_net= NEv4.ChessEvaluator()
    evaluation = new_net(torch.Tensor(getBoard.get_bit_board(board)),torch.Tensor(getBoard.get_info_board(board)))

    move,eval = new_bot.find_best_move(board)
    #print(new_bot.evaluate_board([],board))
    print(move,eval)



