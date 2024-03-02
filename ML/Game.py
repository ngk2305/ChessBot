import chess
import chess.svg
import pygame
import sys
import SimpleNeuralEngine
import time
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
screen_size=800
SQUARE_SIZE = WIDTH // 8
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)

# Initialize chess board


# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")
clock = pygame.time.Clock()

def draw_board():
    for row in range(8):
        for col in range(8):
            square= row+col*8
            color = 'LightSquare1' if (row + col) % 2 == 0 else 'DarkSquare1'
            filename = f"board/{color}.png"
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            screen.blit(piece_image,
                        (chess.square_file(square) * SQUARE_SIZE, (7 - chess.square_rank(square)) * SQUARE_SIZE))
def draw_pieces(board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color:
                side='W'
            else:
                side='B'
            filename = f"pieces/{piece.symbol()+side}.png"  # assuming you have chess piece images in a folder named 'pieces'
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            screen.blit(piece_image, (chess.square_file(square) * SQUARE_SIZE, (7 - chess.square_rank(square)) * SQUARE_SIZE))

def main(wp,bp,depth,custom_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
    board = chess.Board(fen=custom_fen)
    running = True
    dragging = False
    selected_square = None
    taken_moves = []
    bot=SimpleNeuralEngine.Agent(3)
    while running and not board.is_game_over():

        if (board.turn == chess.WHITE and wp) or (board.turn == chess.BLACK and bp):
            print('Human playing')
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    file = x // (screen_size // 8)
                    rank = 7 - y // (screen_size // 8)  # Invert Y-axis to match chess coordinates

                    square = chess.square(file, rank)

                    if selected_square is None:
                        if board.piece_at(square) is not None:
                            selected_square = square
                    else:
                        move = chess.Move(selected_square, square)

                        if move in board.legal_moves:
                            board.push(move)
                            taken_moves = []

                        selected_square = None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if len(board.move_stack) > 0:
                            last_move = board.pop()
                            taken_moves.append(last_move)

                    # Check for redo hotkey (right arrow)
                    elif event.key == pygame.K_RIGHT:
                        if len(taken_moves) > 0:
                            redo_move = taken_moves.pop()
                            board.push(redo_move)
        else:

            move= bot.find_best_move(board)
            print(move)
            board.push(move)
        screen.fill(WHITE)
        draw_board()
        draw_pieces(board)

        if selected_square is not None:
            file, rank = chess.square_file(selected_square), 7 - chess.square_rank(selected_square)
            pygame.draw.rect(screen, GREY, (file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 6)
            for move in board.legal_moves:
                if move.from_square == selected_square:
                    to_file, to_rank = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
                    square_rect = pygame.Rect(to_file * (screen_size // 8), to_rank * (screen_size // 8),
                                              screen_size // 8, screen_size // 8)
                    pygame.draw.rect(screen, GREY, square_rect, 6)  # RGB color for grey
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(0,0,3,'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')