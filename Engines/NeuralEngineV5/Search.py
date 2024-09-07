import chess
import math

def simpleMoveOrder(board,alpha):

    #alpha is the coefficient of decay of move probability in the list

    legal_moves = list(board.legal_moves)
    check = []
    capture = []
    none = []
    for move in legal_moves:
        if board.gives_check(move):
            check.append(move)
        elif board.is_capture(move):
            capture.append(move)
        else:
            none.append(move)
    capture = mvv_lva(capture, board)

    move_list = check + capture + none
    prob_split= [(1-alpha)/(1-math.pow(alpha,len(move_list)))*math.pow(alpha,i) for i in range(len(move_list))]
    return move_list , prob_split

# Incorporating static eval of the position


# Piece values used for the MVV-LVA heuristic
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100  # The king's value is set arbitrarily high
}


def mvv_lva(captures, board):
    # List to hold captures with their evaluation
    evaluated_captures = []
    for move in captures:
        # Get the piece being captured (victim)
        victim_piece = board.piece_at(move.to_square)
        # Get the piece making the capture (aggressor)
        aggressor_piece = board.piece_at(move.from_square)

        if victim_piece and aggressor_piece:
            # Calculate the heuristic value (victim value - aggressor value)
            value_difference = piece_values[victim_piece.piece_type] - piece_values[aggressor_piece.piece_type]
            # Store the move along with its value difference
            evaluated_captures.append((move, value_difference))

    # Sort the captures by the value difference (higher is better)
    evaluated_captures.sort(key=lambda x: x[1], reverse=True)

    # Return the sorted list of moves (only the moves, not the value differences)
    return [move for move, value in evaluated_captures]