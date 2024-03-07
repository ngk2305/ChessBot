import chess
import chess.svg
import chess.engine
from Engines.SimpleNeuralEngine import BitboardExtraction
import torch
from Engines.SimpleNeuralEngine import NeuralNetworkSuper

class Agent():
    def __init__(self):
        self.depth= 3
        self.NN= NeuralNetworkSuper.SuperChessEvaluator()
        self.NN.load_state_dict(torch.load(f'Engines/NeuralEngineV2/super_model_weights.pth'))

    def evaluate_board(self,board):
        evaluation = 0
        if not board.is_checkmate():
            evaluation = self.NN(torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32))

        else:
            if not board.is_stalemate():
                if board.turn:
                    evaluation=0
                else:
                    evaluation=1
        return evaluation

    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):


        if depth <= 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            max_eval = 0
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
            min_eval = 1
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
        legal_moves = self.simpleMoveOrder(board)

        best_move = None
        if board.turn == chess.WHITE:  # Maximizing player (white)
            max_eval = 0
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(board, self.depth - 1, 0, 1, False)
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
                eval = self.minimax_alpha_beta(board, self.depth - 1, 0,1, True)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    if min_eval==0:
                        break

        return best_move,eval

    def simpleMoveOrder(self,board):
        legal_moves = list(board.legal_moves)
        check=[]
        capture=[]
        none=[]
        for move in legal_moves:
            if board.gives_check(move):
                check.append(move)
            elif board.is_capture(move):
                capture.append(move)
            else:
                none.append(move)
        return check+capture+none




if __name__== '__main__':
    board = chess.Board(fen='rnbqkbnr/ppppp1pp/5p2/4P3/8/3B4/PPPP1PPP/RNBQK1NR w KQkq - 1 4')
    evaluation_count=0
    new_bot = Agent()

    new_net= NeuralNetworkSuper.SuperChessEvaluator()
    evaluation = new_net(torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32))



    move,eval = new_bot.find_best_move(board)

    #print(new_bot.evaluate_board([],board))
    print(move,eval)
    print(evaluation_count)


