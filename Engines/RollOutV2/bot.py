import chess
import chess.svg
import chess.engine
from Engines.RollOutV2 import eval as e
from Engines.RollOutV2 import Transposition
from Engines.RollOutV2 import Search
import chess.syzygy


class Agent:
    def __init__(self):
        self.table = Transposition.TranspositionTable()
        self.dummy = 0
        self.tablebase = chess.syzygy.open_tablebase("/Users/test/Documents/GitHub/ChessBot/Engines/RollOutV2/syzygy")

    def evaluate_board(self, board, depth):
        return e.eval(board)


    def minimax_alpha_beta(self, board, probability, alpha, beta, maximizing_player, depth):
        PV = []

        if board.is_game_over():
            if board.is_checkmate():
                if board.turn:
                    return [], None,0 + depth * 0.001
                else:
                    return [], None,1 - depth * 0.001
            else:
                if board.is_stalemate() or board.can_claim_draw() or board.is_insufficient_material():
                    return [], None,0.5

        if len(board.piece_map()) <= 5:
            PV, best_move, eval = e.endgameEval(board, self.tablebase)
            self.table.store(board, 1, eval, "EXACT", best_move, PV)
            return PV, best_move, eval

        if self.table.lookup(board) is not None:
            prob, score, flag, best_move, principle_variation = self.table.lookup(board)
            if probability <= prob:
                return principle_variation, best_move, score
            else:
                PV = principle_variation

        if probability <= 0.000002:
            eval = self.evaluate_board(board, depth)
            self.table.store(board, probability, eval, "EXACT", None, [])
            return [], None, eval

        move_list, prob_list = Search.simpleMoveOrder(board, 0.8)
        if maximizing_player:
            max_eval = -0.1
            best_move = None
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
            if best_move is not None:
                PV.append(best_move)
            self.table.store(board, probability, max_eval, "EXACT", best_move, PV)

            return PV,best_move, max_eval
        else:
            min_eval = 1.1
            best_move = None
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
                    try:
                        for i in range(len(PV) -2, -1, -2):
                            if (PV[i] in move_list) & (PV[i] not in move_tried):
                                Search.ReArrange(move_list, PV[i], move)

                    except:
                        print(PV)
                        print(type(PV))
                beta = min(beta, eval)
                if beta <= alpha:
                    self.table.store(board, new_prob, eval, "BETA", None, PV)
                    board.pop()
                    break  # Prune remaining branches
                board.pop()

            if best_move is not None:
                PV.append(best_move)
            self.table.store(board, probability, min_eval, "EXACT", best_move, PV)
            return PV, best_move, min_eval

    def find_best_move(self, board):
        if board.turn == chess.WHITE:
            PV_dummy, best_move, eval = self.minimax_alpha_beta(board, 1, 0.03, 0.97, True, 0)
            return best_move, eval
        else:
            PV_dummy, best_move, eval = self.minimax_alpha_beta(board, 1, 0.03, 0.97, False, 0)
            return best_move, eval


if __name__ == '__main__':
    custom_fen = 'r1r2bk1/3B1p1p/3p2p1/3P2P1/5p1P/Pp1N1P2/1PqQ4/K2RR3 b - - 0 1'
    board = chess.Board(fen=custom_fen)
    bot = Agent()



    while not board.is_game_over():
        move, evaluation = bot.find_best_move(board)
        print(move, evaluation)
        board.push(move)


