import chess

# Create a chess board

board= chess.Board()
# Make some moves (for demonstration purposes)
board.push(chess.Move.from_uci("e2e4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g1f3"))

# Extract bitboards for different pieces
pawn_bitboard = board.pieces(chess.PAWN, chess.WHITE)
knight_bitboard = board.pieces(chess.KNIGHT, chess.WHITE)
bishop_bitboard = board.pieces(chess.BISHOP, chess.WHITE)
rook_bitboard = board.pieces(chess.ROOK, chess.WHITE)
queen_bitboard = board.pieces(chess.QUEEN, chess.WHITE)
king_bitboard = board.pieces(chess.KING, chess.WHITE)

# Custom function to print bitboards
def print_bitboard(bitboard):
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = '1' if bitboard & 1 << square else '0'
            print(piece, end=' ')
        print()

def split_bits(n):
    return [(n >> i) & 1 for i in range(63, -1, -1)]
def get_bit_board(board):
    input=[]
    for i in chess.COLORS:
        for j in chess.PIECE_TYPES:
            bitboard = split_bits(board.pieces(j,i))
            input.append(bitboard)
    return input

def get_bit_fen(fen):
    return get_bit_board(chess.Board(fen=fen))

def get_bit_fen_batch(batch):
    output=[]

    for i in batch:

        output.append(get_bit_fen(i))

    return output

if __name__=='__main__':
    board=chess.Board(fen='r4rk1/1p2qppp/8/pb1PN3/3Q2n1/P1R5/1P1N2BP/R5K1 b - - 0 26')
    print(get_bit_board(board))
    print(get_bit_fen('r4rk1/1p2qppp/8/pb1PN3/3Q2n1/P1R5/1P1N2BP/R5K1 b - - 0 26'))
