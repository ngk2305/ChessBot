import chess

def get_info_board(board):
    xtra = []
    xtra.append(int(board.turn))
    xtra.append(int(bool(board.castling_rights & chess.BB_H1)))
    xtra.append(int(bool(board.castling_rights & chess.BB_A1)))
    xtra.append(int(bool(board.castling_rights & chess.BB_H8)))
    xtra.append(int(bool(board.castling_rights & chess.BB_A8)))

    return xtra
if __name__ == '__main__':
    print(get_info_board(chess.Board(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')))

