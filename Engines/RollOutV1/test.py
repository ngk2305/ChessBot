import chess

# Create a chess board
board = chess.Board()

# Material counting dictionary
material_count = {
    'P': 0,  # White pawns
    'N': 0,  # White knights
    'B': 0,  # White bishops
    'R': 0,  # White rooks
    'Q': 0,  # White queens
    'K': 0,  # White king
    'p': 0,  # Black pawns
    'n': 0,  # Black knights
    'b': 0,  # Black bishops
    'r': 0,  # Black rooks
    'q': 0,  # Black queens
    'k': 0,  # Black king
}

# Iterate over all squares of the board
for square in chess.SQUARES:
    piece = board.piece_at(square)

    if piece:
        print(piece.piece_type)
        print(piece.color)

# Print the material count
print(material_count)