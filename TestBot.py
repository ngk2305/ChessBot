import chess
import chess.svg
import chess.engine
from Engines.SimpleNeuralEngine import BitboardExtraction
import torch
from Engines.SimpleNeuralEngine import NeuralNetworkSuper

class Agent():
    def __init__(self):
        self.depth= 5
        self.NN= NeuralNetworkSuper.SuperChessEvaluator()
        self.NN.load_state_dict(torch.load(f'Engines/SimpleNeuralEngine/super_model_weights.pth'))

    def evaluate_board(self,move_taken,board):
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

    def minimax_alpha_beta(self,move_taken ,board, depth, alpha, beta, maximizing_player):


        if depth == 0 or board.is_game_over():
            return self.evaluate_board(move_taken,board)

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                move_taken.append(move)
                eval = self.minimax_alpha_beta(move_taken,board, depth - 1, alpha, beta, False)
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
                move_taken.append(move)
                eval = self.minimax_alpha_beta(move_taken,board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Prune remaining branches
            return min_eval

    def find_best_move(self, board):
        legal_moves = list(board.legal_moves)
        move_taken=[]
        eval_list=[]
        best_move = None
        if board.turn == chess.WHITE:  # Maximizing player (white)
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                move_taken.append(move)
                eval = self.minimax_alpha_beta(move_taken,board, self.depth - 1, float('-inf'), float('inf'), False)
                eval_list.append([move,eval])
                board.pop()
                if eval > max_eval:
                    print('new eval is')
                    print(eval)
                    print('old eval is')
                    print(max_eval)
                    max_eval = eval
                    best_move = move
            print(eval_list)
            eval=max_eval
        else:  # Minimizing player (black)
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax_alpha_beta(move_taken,board, self.depth - 1, float('-inf'), float('inf'), True)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            eval = min_eval
        return best_move,eval

if __name__== '__main__':
    board = chess.Board(fen='rnbqkbnr/ppppp1pp/5p2/4P3/8/3B4/PPPP1PPP/RNBQK1NR w KQkq - 1 4')
    new_bot = Agent()

    new_net= NeuralNetworkSuper.SuperChessEvaluator()
    evaluation = new_net(torch.tensor(BitboardExtraction.get_bit_board(board), dtype=torch.float32))



    move,eval = new_bot.find_best_move(board)

    #print(new_bot.evaluate_board([],board))
    print(move,eval)


