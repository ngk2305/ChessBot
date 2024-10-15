import chess
import eval
def get_attacked_pieces(board):
    attacked_pieces = []
    for square in chess.SQUARES:
        if board.piece_at(square):
            if not (board.piece_at(square).color ==(not board.turn)) and board.is_attacked_by(not board.turn, square):
                attacked_pieces.append((square, board.piece_at(square)))
    return attacked_pieces

def if_make_new_attack(move,board):
    new_attack_detected = False
    enemy_attacked_before = get_attacked_pieces(board)
    board.push(move)
    enemy_attacked_after = get_attacked_pieces(board)
    board.pop()
    for attack in enemy_attacked_after:
        if attack not in enemy_attacked_before:
            new_attack_detected = True
    return new_attack_detected

def move_improve_static_eval(move,board):
    old_eval = eval.eval(board)
    board.push(move)
    new_eval = eval.eval(board)
    board.pop()
    if new_eval > old_eval:
        return True
    else:
        return False

def get_move_type(board, move):
    if board.gives_check(move):
        return 4
    elif board.is_capture(move):
        return 3
    elif if_make_new_attack(move,board):
        return 2
    elif move_improve_static_eval(move,board):
        return 1
    else:
        return 0


if __name__ == '__main__':
    board = chess.Board(fen='r1bqkbnr/pppppppp/2n5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2')
    for move in list(board.legal_moves):
        print(move)
        print(get_move_type(board,move))