import chess
import torch
# Create a chess board


def split_bits(n):
    return [(n >> i) & 1 for i in range(63, -1, -1)]


def get_bit_board(board):
    piece_map = []
    for j in chess.PIECE_TYPES:

        bitboard_white = split_bits(board.pieces(j, True))
        bitboard_black = split_bits(board.pieces(j, False))

        list_64 = [float(elementW)-float(elementB) for elementW, elementB in zip(bitboard_white, bitboard_black)]
        list8x8 = [list_64[i:i+8] for i in range(0, len(list_64), 8)]

        piece_map.append(list8x8)

    return piece_map


def get_bit_fen(fen):
    return get_bit_board(chess.Board(fen=fen))


def get_bit_fen_batch(batch):
    output=[]

    for i in batch:

        output.append(get_bit_fen(i))

    return output


def get_dimensions(nested_list):
    def get_shape(lst):
        if not isinstance(lst, list):
            return []
        return [len(lst)] + get_shape(lst[0])

    return get_shape(nested_list)


def get_info_board(board):
    xtra = []
    xtra.append(float(board.turn))
    xtra.append(float(bool(board.castling_rights & chess.BB_H1)))
    xtra.append(float(bool(board.castling_rights & chess.BB_A1)))
    xtra.append(float(bool(board.castling_rights & chess.BB_H8)))
    xtra.append(float(bool(board.castling_rights & chess.BB_A8)))

    return xtra

if __name__=='__main__':
    board=chess.Board(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    print(get_dimensions(get_bit_board(board)))
    print(torch.Tensor(get_bit_board(board)))
    print(get_bit_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))
    print(torch.Tensor(get_info_board(chess.Board(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))))


