def fen_to_bitboard(fen):
    # Initialize bitboards for each piece type
    bitboards = {
        'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
        'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
    }
    bitboard_all = 0

    # Parsing FEN string
    parts = fen.split()
    ranks = parts[0].split('/')
    for rank_index, rank in enumerate(ranks):
        file_index = 0
        for char in rank:
            if char.isdigit():
                file_index += int(char)
            else:
                piece_type = char
                # Calculate bitboard index
                square_index = 7 - file_index + (rank_index * 8)
                # Set the corresponding bit
                bitboards[piece_type] |= 1 << square_index
                bitboard_all |= 1 << square_index
                file_index += 1

    return bitboards, bitboard_all

# Example usage:
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
bitboards, bitboard_all = fen_to_bitboard(fen)
print("Bitboards for each piece type:")
print(bitboards)
print("\nBitboard for all pieces:")
print(bin(bitboard_all))