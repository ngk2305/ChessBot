import chess
import chess.svg
import pygame
import sys
from UI import moveIndicator
from UI import screenPieceBoard

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)


def main(wp,bp,depth,custom_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
    #Init pygame, get screen/clock/FPS



    screen,clock, FPS= screenPieceBoard.get_screen_clock()

    #Game variable
    board = chess.Board(fen=custom_fen)
    running = True
    selected_square = None
    taken_moves = []
    white_bot= None
    black_bot= None
    while running and not board.is_game_over():

        if (board.turn == chess.WHITE and wp) or (board.turn == chess.BLACK and bp):
            selected_square,taken_moves,running = moveIndicator.event_scan(board,selected_square,taken_moves,running)

        else:
            print('Bot is thinking...')
            if board.turn:
                move = white_bot.find_best_move(board)
            else:
                move = black_bot.find_best_move(board)

            board.push(move)


        screen.fill(WHITE)
        screenPieceBoard.update_display(screen,board,clock,selected_square)


    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(1,1,3,'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')