import pygame
import chess

# Constants
WIDTH, HEIGHT = 800, 800
SCREEN_SIZE=800
SQUARE_SIZE = WIDTH // 8
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)

def event_scan(board,selected_square,taken_moves,running):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            file = x // (SCREEN_SIZE // 8)
            rank = 7 - y // (SCREEN_SIZE // 8)  # Invert Y-axis to match chess coordinates

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

    return selected_square, taken_moves, running

def show_piece_move(board,selected_square,screen):

    if selected_square is not None:
        file, rank = chess.square_file(selected_square), 7 - chess.square_rank(selected_square)
        pygame.draw.rect(screen, GREY, (file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 6)
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_file, to_rank = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
                square_rect = pygame.Rect(to_file * (SCREEN_SIZE // 8), to_rank * (SCREEN_SIZE // 8),
                                          SCREEN_SIZE // 8, SCREEN_SIZE // 8)
                pygame.draw.rect(screen, GREY, square_rect, 6)  # RGB color for grey
