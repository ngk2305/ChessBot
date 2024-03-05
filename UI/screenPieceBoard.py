import pygame
import chess
from UI import moveIndicator
# Constants
WIDTH, HEIGHT = 800, 800
screen_size=800
SQUARE_SIZE = WIDTH // 8
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)

def draw_board(screen):

    for row in range(8):
        for col in range(8):
            square= row+col*8
            color = 'LightSquare1' if (row + col) % 2 == 1 else 'DarkSquare1'
            filename = f"UI/TexturePack/board/{color}.png"
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            screen.blit(piece_image,
                        (chess.square_file(square) * SQUARE_SIZE, (7 - chess.square_rank(square)) * SQUARE_SIZE))

def draw_pieces(screen,board):

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color:
                side='W'
            else:
                side='B'
            filename = f"UI/TexturePack/pieces/{piece.symbol()+side}.png"  # assuming you have chess piece images in a folder named 'pieces'
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            screen.blit(piece_image, (chess.square_file(square) * SQUARE_SIZE, (7 - chess.square_rank(square)) * SQUARE_SIZE))

def get_screen_clock():
    pygame.init()
    screen= pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    FPS=15
    return screen,clock,FPS

def update_display(screen,board,clock,selected_square):

    draw_board(screen)
    draw_pieces(screen,board)
    moveIndicator.show_piece_move(board,selected_square,screen)
    pygame.display.flip()
    clock.tick(FPS)