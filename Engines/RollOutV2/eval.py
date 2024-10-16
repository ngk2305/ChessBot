import chess
import time
import math
import chess.syzygy
import sys

DOUBLED_PAWN_PENALTY = 10
ISOLATED_PAWN_PENALTY = 20
BACKWARDS_PAWN_PENALTY = 8
PASSED_PAWN_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
ROOK_OPEN_FILE_BONUS = 15
ROOK_ON_SEVENTH_BONUS = 20

piece_value = [0, 100, 300, 315, 450, 900, 0]

flip = [
	 56,  57,  58,  59,  60,  61,  62,  63,
	 48,  49,  50,  51,  52,  53,  54,  55,
	 40,  41,  42,  43,  44,  45,  46,  47,
	 32,  33,  34,  35,  36,  37,  38,  39,
	 24,  25,  26,  27,  28,  29,  30,  31,
	 16,  17,  18,  19,  20,  21,  22,  23,
	  8,   9,  10,  11,  12,  13,  14,  15,
	  0,   1,   2,   3,   4,   5,   6,   7
	]

pawn_pcsq = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 15, 20, 20, 15, 10, 5,
    4, 8, 12, 16, 16, 12, 8, 4,
    3, 6, 9, 12, 12, 9, 6, 3,
    2, 4, 6, 8, 8, 6, 4, 2,
    1, 2, 3, -10, -10, 3, 2, 1,
    0, 0, 0, -40, -40, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
]

knight_pcsq = [
	-10, -10, -10, -10, -10, -10, -10, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10,   0,   5,   5,   5,   5,   0, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   0,   5,   5,   5,   5,   0, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10, -30, -10, -10, -10, -10, -30, -10
]

bishop_pcsq = [
	-10, -10, -10, -10, -10, -10, -10, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10,   0,   5,   5,   5,   5,   0, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   0,   5,   5,   5,   5,   0, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10, -10, -20, -10, -10, -20, -10, -10
]

king_pcsq = [
	-40, -40, -40, -40, -40, -40, -40, -40,
	-40, -40, -40, -40, -40, -40, -40, -40,
	-40, -40, -40, -40, -40, -40, -40, -40,
	-40, -40, -40, -40, -40, -40, -40, -40,
	-40, -40, -40, -40, -40, -40, -40, -40,
	-40, -40, -40, -40, -40, -40, -40, -40,
	-20, -20, -20, -20, -20, -20, -20, -20,
	  0,  20,  40, -20,   0, -20,  40,  20
]

king_endgame_pcsq = [
	  0,  10,  20,  30,  30,  20,  10,   0,
	 10,  20,  30,  40,  40,  30,  20,  10,
	 20,  30,  40,  50,  50,  40,  30,  20,
	 30,  40,  50,  60,  60,  50,  40,  30,
	 30,  40,  50,  60,  60,  50,  40,  30,
	 20,  30,  40,  50,  50,  40,  30,  2 	,
	 10,  20,  30,  40,  40,  30,  20,  10,
	  0,  10,  20,  30,  30,  20,  10,   0
]


def color_to_index(color):
	#Convert White (True) to 0 and Black (False) to 1
	if color:
		return 0
	else:
		return 1


def eval(board):
	#f to keep track highest pawn rank
	f = 0
	#pawn rank keep tracks of the least advanced pawn in each file
	pawn_rank = [[7 for i in range(10)], [0 for j in range(10)]]
	piece_mat = [0 for k in range(2)]
	pawn_mat = [0 for l in range(2)]

	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			if piece.piece_type == 1:
				pawn_mat[color_to_index(piece.color)] += piece_value[1]
				f = chess.square_file(square) + 1
				if piece.color:
					if pawn_rank[0][f] > chess.square_rank(square):
						pawn_rank[0][f] = chess.square_rank(square)
				else:
					if pawn_rank[1][f] < chess.square_rank(square):
						pawn_rank[1][f] = chess.square_rank(square)
			else:
				piece_mat[color_to_index(piece.color)] += piece_value[piece.piece_type]

	#Second pass:evaluate each piece

	score = [0, 0]
	score[0] = piece_mat[0] + pawn_mat[0]
	score[1] = piece_mat[1] + pawn_mat[1]



	for square in chess.SQUARES:
		piece = board.piece_at(square)
		if piece:
			if piece.color:
				match piece.piece_type:
					case 1:
						score[0] += eval_light_pawn(square, pawn_rank)

					case 2:
						score[0] += knight_pcsq[flip[square]]

					case 3:
						score[0] += bishop_pcsq[flip[square]]

					case 4:
						if pawn_rank[0][chess.square_file(square)+1] == 7:
							if pawn_rank[1][chess.square_file(square)+1] ==0:
								score[0] += ROOK_OPEN_FILE_BONUS
							else:
								score[0] += ROOK_SEMI_OPEN_FILE_BONUS
						if chess.square_rank(square) == 6:
							score[0] += ROOK_ON_SEVENTH_BONUS

					case 6:
						if piece_mat[1] <= 1200:
							score[0] += king_endgame_pcsq[flip[square]]
						else:
							score[0] += king_pcsq[flip[square]]



			else:
				match piece.piece_type:
					case 1:
						score[1] += eval_dark_pawn(square, pawn_rank)

					case 2:
						score[1] += knight_pcsq[square]

					case 3:
						score[1] += bishop_pcsq[square]

					case 4:
						if pawn_rank[1][chess.square_file(square)+1] == 0:
							if pawn_rank[0][chess.square_file(square)+1] == 7:
								score[1] += ROOK_OPEN_FILE_BONUS
							else:
								score[1] += ROOK_SEMI_OPEN_FILE_BONUS
						if chess.square_rank(square) == 1:
							score[1] += ROOK_ON_SEVENTH_BONUS

					case 6:
						if piece_mat[0] <= 1200:
							score[1] += king_endgame_pcsq[square]
						else:
							score[1] += king_pcsq[square]



	return eval_rescale(score[0]-score[1])


def eval_light_pawn(i, pawn_rank):

	r = 0
	f = chess.square_file(i)+1

	r+= pawn_pcsq[flip[i]]

	if pawn_rank[0][f] < chess.square_rank(i):
		r -= DOUBLED_PAWN_PENALTY


	if (pawn_rank[0][f-1] == 7) & (pawn_rank[0][f+1] == 7):
		r -= ISOLATED_PAWN_PENALTY
	elif (pawn_rank[0][f - 1] > chess.square_rank(i)) & (pawn_rank[0][f + 1] > chess.square_rank(i)):
		r -= BACKWARDS_PAWN_PENALTY

	if (pawn_rank[1][f-1] <= chess.square_rank(i)) & (pawn_rank[1][f] <= chess.square_rank(i)) & (pawn_rank[1][f+1] <= chess.square_rank(i)):
		r += chess.square_rank(i) * PASSED_PAWN_BONUS

	return r


def eval_dark_pawn(i, pawn_rank):

	r = 0
	f = chess.square_file(i)+1

	r += pawn_pcsq[i]

	if pawn_rank[1][f] > chess.square_rank(i):
		r -= DOUBLED_PAWN_PENALTY


	if (pawn_rank[1][f-1] == 0) & (pawn_rank[1][f+1] == 0):
		r -= ISOLATED_PAWN_PENALTY
	elif (pawn_rank[1][f - 1] < chess.square_rank(i)) & (pawn_rank[1][f + 1] < chess.square_rank(i)):
		r -= BACKWARDS_PAWN_PENALTY

	if (pawn_rank[0][f-1] >= chess.square_rank(i)) & (pawn_rank[0][f] >= chess.square_rank(i)) & (pawn_rank[0][f+1] >= chess.square_rank(i)):
		r += (7-chess.square_rank(i)) * PASSED_PAWN_BONUS

	return r

def eval_rescale(evaluation):
	evaluation = int(evaluation) / 100
	score = 1 / (1 + math.exp(-0.8 * evaluation))
	return score*0.94+0.03

def findBest(board, tablebase):
		if tablebase.probe_dtz(board) != 1:
			legal_moves = list(board.legal_moves)

			best_move = legal_moves[0]

			best_eval = 100
			for move in legal_moves:
				board.push(move)
				try:
					eval = 1/tablebase.probe_dtz(board)
				except:
					eval = 0
				if (eval < best_eval) :
					best_eval = eval
					best_move = move
				board.pop()
			return best_move
		else:
			legal_moves = list(board.legal_moves)
			best_move = legal_moves[0]
			best_eval = 100
			for move in legal_moves:
				board.push(move)
				try:
					eval = 1 / tablebase.probe_dtz(board)
				except:
					eval = 0
				if eval < best_eval and (eval != (-0.5) or board.is_capture(move)):
					best_eval = eval
					best_move = move
				board.pop()
			return best_move


def endgameEval(og_board, tablebase):
		PV=[]
		board = og_board

		if tablebase.probe_dtz(board) == 0:
			i = 0
			while (i<20) and (not board.is_game_over()):
				best_move = findBest(board, tablebase)
				PV.append(best_move)
				board.push(best_move)
				i+=1

			for j in range(i):
				board.pop()
			PV = PV[::-1]
			return PV, findBest(board, tablebase), 0.5
		else:
			result = 0
			counter = 0
			if (board.turn and tablebase.probe_dtz(board) > 0) or ((not board.turn) and tablebase.probe_dtz(board) < 0):
				result = 1
			best = findBest(board, tablebase)
			while not board.is_game_over():
				best_move = findBest(board, tablebase)
				PV.append(best_move)
				board.push(best_move)
				counter+=1
			for i in range(counter):
				board.pop()
			PV = PV[::-1]
			return PV, best, result

if __name__ == '__main__':
	#tablebase = chess.syzygy.open_tablebase("/Users/test/Documents/GitHub/ChessBot/Engines/RollOutV2/syzygy")
	#board = chess.Board(fen = '8/4kq2/7K/7P/6P1/8/8/8 w - - 0 1')
	#print(endgameEval(board, tablebase))
	board = chess.Board(fen = '8/RR6/2k1p1p1/2p2b1p/5P2/4K1P1/P6P/3r4 w - - 3 35')
	print(eval(board))

