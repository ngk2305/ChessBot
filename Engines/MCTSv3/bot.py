import chess
import threading
import MCNet
import torch
import NeuralMCTS

class Agent:
    def __init__(self, version, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.net = MCNet.ChessEvaluator()
        if version == 'current':
            try:
                self.net.load_state_dict(torch.load('epoch4.pth', map_location=torch.device('cpu')))
            except:
                print("Current not found")
        elif version == 'best':
            try:
                self.net.load_state_dict(torch.load(f'Engines/NeuralEngineV5/NEv5_weights.pth'))
            except:
                print("Best not found")

        self.node = NeuralMCTS.Node(None,self.net.forward)

        self.board = chess.Board()
        self.starting_fen = fen
        self.searching = True

    def find_best_move(self):
        t = threading.Thread(target=controlled_loop, args=(control_flag,))
        self.node = self.node.search(100,20,self.starting_fen,self.searching)
        board, eval = self.node.get_move_eval()
        print(f"bestmove {board}")
        return board

    def uci_mode(self):
        print('MCTSv2')
        print('Gia Khanh')
        print('uciok')

    def parse_uci_command(self,command):
        if command == 'uci':
            self.uci_mode()
        elif command == 'isready':
            self.handle_isready()
        elif command == 'ucinewgame':
            self.start_new_game()
        elif command.startswith('position'):
            self.handle_position(command)
        elif command.startswith('go'):
            self.find_best_move()
        elif command == 'stop':
            self.stop_search()
        elif command == 'quit':
            self.quit_game()

    def stop_search(self):
        self.node.set_searching(False)

    def start_new_game(self):

        self.node = NeuralMCTS.Node(None, self.net.forward)

        self.board = chess.Board()
        self.starting_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    def handle_position(self,command):
        if 'startpos' in command:
            # Reset the board to the initial position
            self.board = chess.Board()
            if 'moves' in command:
                moves = command.split('moves ')[1].split()  # Get the list of moves
                for move in moves:
                    self.board.push(move)  # Apply each move to the board
        if 'fen' in command:
            fen_part = command.split('fen ')[1].split(' moves ')[0]  # Get FEN string
            self.board = chess.Board(fen=fen_part)  # Set the position from the FEN
            if 'moves' in command:
                moves = command.split('moves ')[1].split()  # Get the list of moves
                for move in moves:
                    self.board.push(move)  # Apply each move to the board

    def handle_isready(self):
        print("readyok")

    def quit_game(self):
        exit(0)

if __name__ == '__main__':
    bot = Agent('current','2r3k1/p4p2/3Rp2p/1p2PqpK/8/1P4P1/P3Q2P/8 b - - 4 3')

    print(bot.find_best_move())


