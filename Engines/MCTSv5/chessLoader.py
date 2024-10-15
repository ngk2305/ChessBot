from torch.utils.data import Dataset
import chess
import torch
import getBoard

class ChessDataset(Dataset):
    def __init__(self, games_data):
        self.games_data = games_data


    def __len__(self):
        # Total number of plies across all games
        return sum(len(game['moves']) for game in self.games_data)

    def __getitem__(self, idx):
        # Find which game and ply the index corresponds to
        current_idx = 0
        for game in self.games_data:
            if current_idx + len(game['moves']) > idx:
                move_idx = idx - current_idx
                return self.get_ply_data(game, move_idx)
            current_idx += len(game['moves'])

    def get_ply_data(self, game, move_idx):
        board = chess.Board()

        # Play up to the current move
        for i, move in enumerate(game['moves'][:move_idx]):
            board.push_san(move)

        # Get board state
        board_matrix = torch.Tensor(getBoard.get_bit_board(board)),torch.Tensor(getBoard.get_info_board(board))

        # Get the move made at the current move
        move = game['moves'][move_idx]
        move = board.parse_san(move).uci()
        move = chess.Move.from_uci(move)

        move_from_to = move.from_square, move.to_square

        # Get evaluation (can be the game's result or an external evaluation function)

        return board_matrix, torch.tensor(move_from_to)