import pickle

import chess

import MCNet
import torch
import NeuralMCTS

class Agent:
    def __init__(self, version, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.net = MCNet.ChessEvaluator()
        if version == 'current':
            try:
                self.net.load_state_dict(torch.load('current.pth'))
            except:
                print("Current not found")
        elif version == 'best':
            try:
                self.net.load_state_dict(torch.load(f'Engines/NeuralEngineV5/NEv5_weights.pth'))
            except:
                print("Best not found")

        self.node = NeuralMCTS.Node(None,self.net.forward)


        self.starting_fen = fen

    def find_best_move(self):
        self.node = self.node.search(10,self.starting_fen)

        return self.node.get_move_eval()

if __name__ == '__main__':
    bot = Agent('current','2r3k1/p4p2/3Rp1qp/1p2P1p1/6K1/1P4P1/P3Q2P/8 b - - 2 2')

    print(bot.find_best_move())


