import BitboardExtraction
import XtraInfo
import chess
import torch

def fullInfo(fen):
    board = chess.Board(fen=fen)
    return torch.Tensor(BitboardExtraction.get_bit_board(board)),torch.Tensor(XtraInfo.get_info_board(board))

if __name__ == '__main__':
    x = fullInfo('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    print(x)
