import chess
import chess.syzygy

def findBest(board, tablebase):
    if tablebase.probe_dtz(board) != 1:
        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = 100
        print(legal_moves)
        for move in legal_moves:
            board.push(move)
            try:
                eval = 1 / tablebase.probe_dtz(board)
                print(move)
                print(eval)
            except:
                eval = 0
            if (eval < best_eval):
                best_eval = eval
                best_move = move
            board.pop()
        return best_move
    else:
        legal_moves = list(board.legal_moves)
        best_move = legal_moves[0]
        best_eval = 100
        for move in legal_moves:
            board.push(move)
            try:
                eval = 1 / tablebase.probe_dtz(board)
                print(move)
                print(eval)
            except:
                eval = 0
            if eval < best_eval and (eval != (-0.5) or board.is_capture(move)):
                best_eval = eval
                best_move = move
            board.pop()
        return best_move


def endgameEval(og_board, tablebase):
    PV = []
    board = og_board

    if tablebase.probe_dtz(board) == 0:
        for i in range(20):
            best_move = findBest(board, tablebase)
            PV.append(best_move)
            board.push(best_move)

        for j in range(20):
            board.pop()
        PV = PV[::-1]
        return PV, findBest(board, tablebase), 0.5
    else:
        result = 0
        counter = 0
        if (board.turn and tablebase.probe_dtz(board) > 0) or ((not board.turn) and tablebase.probe_dtz(board) < 0):
            result = 1
        best = findBest(board, tablebase)
        while not board.is_game_over():
            best_move = findBest(board, tablebase)
            PV.append(best_move)
            board.push(best_move)
            counter += 1
        for i in range(counter):
            board.pop()
        PV = PV[::-1]
        return PV, best, result

if __name__ == '__main__':
    board = chess.Board(fen='8/4kq2/6PK/7P/8/8/8/8 b - - 0 4')
    tablebase = chess.syzygy.open_tablebase("/Users/test/Documents/GitHub/ChessBot/Engines/RollOutV2/syzygy")
    print(findBest(board, tablebase))