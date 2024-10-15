import chess
import chess.engine

# Path to the AVX2 version of Stockfish binary
STOCKFISH_PATH = "/path/to/your/stockfish-avx2"

# Initialize the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Create a chess board
board = chess.Board()

# Let Stockfish play the best move
result = engine.play(board, chess.engine.Limit(time=2.0))  # Think for 2 seconds
print(f"Stockfish suggests move: {result.move}")

# You can also get the evaluation of the position
info = engine.analyse(board, chess.engine.Limit(depth=20))
print(f"Evaluation: {info['score']}")

# Quit the engine
engine.quit()