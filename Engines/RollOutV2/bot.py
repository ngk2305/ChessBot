import chess
import chess.svg
import chess.engine
from Engines.RollOutV2 import eval
from Engines.RollOutV2 import Transposition
from Engines.RollOutV2 import Search


class Agent:
    def __init__(self):
        self.table = Transposition.TranspositionTable()
        self.dummy = 0

    def evaluate_board(self, board, depth):
        if not board.is_checkmate():
            return eval.eval(board)
        else:
            if not board.is_stalemate():
                if board.turn:
                    return 0+depth*0.001
                else:
                    return 1-depth*0.001


    def minimax_alpha_beta(self, board, probability, alpha, beta, maximizing_player, depth):

        if self.table.lookup(board) is not None:
            prob, score, flag, best_move, principle_variation = self.table.lookup(board)
            if probability <= prob:
                return principle_variation, best_move, score

        if probability <= 0.000002 or board.is_game_over():
            eval = self.evaluate_board(board, depth)
            self.table.store(board, probability, eval, "EXACT", None, [])
            return [], None, eval

        move_list, prob_list = Search.simpleMoveOrder(board, 0.8)
        if maximizing_player:
            max_eval = -0.1
            best_move = None
            PV = []
            move_tried = []
            for move, prob in zip(move_list, prob_list):
                move_tried.append(move)
                board.push(move)
                new_prob = probability * prob
                PV_temp, dummy_move, eval = self.minimax_alpha_beta(board, new_prob, alpha, beta, False, depth+1)

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    PV = PV_temp
                    for i in range(len(PV) -2, -1, -2):
                        if (PV[i] in move_list) & (PV[i] not in move_tried):
                            Search.ReArrange(move_list, PV[i], move)


                alpha = max(alpha, eval)
                if beta <= alpha:
                    self.table.store(board, new_prob , eval, "ALPHA", None, PV)
                    board.pop()
                    break  # Prune remaining branches



                board.pop()
            PV.append(best_move)
            self.table.store(board, probability, max_eval, "EXACT", best_move, PV)

            return PV,best_move, max_eval
        else:
            min_eval = 1.1
            best_move = None
            PV = []
            move_tried = []
            for move, prob in zip(move_list, prob_list):
                move_tried.append(move)
                board.push(move)
                new_prob = probability * prob
                PV_temp , dummy_move, eval = self.minimax_alpha_beta(board, new_prob, alpha, beta, True, depth+1)

                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                    PV = PV_temp
                    for i in range(len(PV) -2, -1, -2):
                        if (PV[i] in move_list) & (PV[i] not in move_tried):
                            Search.ReArrange(move_list, PV[i], move)

                beta = min(beta, eval)
                if beta <= alpha:
                    self.table.store(board, new_prob, eval, "BETA", None, PV)
                    board.pop()
                    break  # Prune remaining branches
                board.pop()

            PV.append(best_move)
            self.table.store(board, probability, min_eval, "EXACT", best_move, PV)
            return PV, best_move, min_eval

    def find_best_move(self, board):
        if board.turn == chess.WHITE:
            return self.minimax_alpha_beta(board, 1, 0, 1, True, 0)
        else:
            return self.minimax_alpha_beta(board, 1, 0, 1, False, 0)


if __name__ == '__main__':
    custom_fen = '6k1/3qp1b1/p2p2p1/5B2/4nP2/4B2Q/2P5/1K4RR b - - 0 29'
    board = chess.Board(fen=custom_fen)
    bot = Agent()


    while not board.is_game_over():
        PV,move, evaluation = bot.find_best_move(board)
        print(PV, move, evaluation)
        board.push(move)
