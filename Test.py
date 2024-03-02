import chess
import chess.svg


def print_board(board):
    print(board)


def print_legal_moves(board):
    print("Legal Moves:")
    for move in board.legal_moves:
        print(move.uci(), end=" ")
    print()


def main():
    board = chess.Board()

    while not board.is_game_over():
        print_board(board)
        print_legal_moves(board)

        move_uci = input("Enter your move (in UCI format, e.g., e2e4): ")

        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid move format. Please use UCI format (e.g., e2e4).")

    print("Game Over")
    print("Result: " + board.result())


if __name__ == "__main__":
    main()
