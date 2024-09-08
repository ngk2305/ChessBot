import chess

from UI import moveIndicator
from UI import screenPieceBoard
import LoadBot


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)


def main(custom_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):


    white_bot,wp = LoadBot.Load_bot('w')
    black_bot,bp = LoadBot.Load_bot('b')



    #Game variable
    board = chess.Board(fen=custom_fen)
    running = True
    selected_square = None
    taken_moves = []


    while running and not board.is_game_over():
        print('Bot is thinking...')
        if board.turn:
            move,eval = white_bot.find_best_move(board)
        else:
            move,eval = black_bot.find_best_move(board)
        print(move,eval)
        board.push(move)

        if board.can_claim_draw() or board.is_seventyfive_moves() or board.is_insufficient_material():
            print("Draw! Game over.")
            break





if __name__ == "__main__":
    #main('8/k3P3/8/8/8/8/8/4K3 w - - 0 1')
    main()