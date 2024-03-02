import chess
import chess.svg
import chess.engine
import NeuralNetwork
import BitboardExtraction
import torch
import NeuralNetworkSuper

class Agent():
    def __init__(self,depth):
        self.depth= depth
        self.NN= NeuralNetworkSuper.SuperChessEvaluator()
        self.NN.load_state_dict(torch.load('super_model_weights.pth'))

    def evaluate_board(self,board):
        evaluation = 0
        if not board.is_checkmate():
            evaluation = self.NN(torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32))

        else:
            if not board.is_stalemate():
                if board.turn:
                    evaluation=1
                else:
                    evaluation=-1
        return evaluation

    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):


        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Prune remaining branches
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Prune remaining branches
            return min_eval

    def find_best_move(self, board):
        legal_moves = list(board.legal_moves)

        best_move = None
        if board.turn == chess.WHITE:  # Maximizing player (white)
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, self.depth - 1, float('-inf'), float('inf'), False)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
        else:  # Minimizing player (black)
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, self.depth - 1, float('-inf'), float('inf'), True)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move

        return best_move

if __name__== '__main__':
    board = chess.Board()
    new_bot = Agent(1)

    new_net= NeuralNetwork.ChessEvaluator()
    evaluation = new_net(torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32))


    move = new_bot.find_best_move(board)


