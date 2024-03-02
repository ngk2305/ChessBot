import chess
import SimpleEngine

custom_fen= '3k4/8/8/8/8/8/8/3KQ3 w - - 0 1'
board= chess.Board(fen=custom_fen)

bot= SimpleEngine.Agent(5)
eva= bot.evaluate_board(board)
print(eva)
print(bot.find_best_move(board,2))