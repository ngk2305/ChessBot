import math
import random
import pickle
import eval
import chess
from tqdm import tqdm
import getBoard
import torch
import chess_utils
from MCNet import ChessEvaluator

class Node:
    def __init__(self, board, nn_pred, parent=None):
        self.board = board  # Current state of the game
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.score = 0  # Number of wins from this node
        self.nn_predictor = nn_pred

    def add_child(self, child_state):
        child_node = Node(child_state, self.nn_predictor, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.score += result

    def UCB1(self, exploration_param=0.125):
        if self.visits == 0:
            return math.inf  # Assign infinite value if never visited
        return (self.score / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def search(self, num_simulations, fen):
        for _ in tqdm(range(num_simulations)):
            node = self._select(self, fen)
            result = node._simulate(fen)
            self._backpropagate(node, result)

        for child in self.children:
            print(child.board)
            print(child.score/child.visits)
            print(child.visits)
        # Randomly select a child based on the weights

        return max(self.children, key=lambda child: child.visits)

    def _select(self, node, fen):
        while node.children:
            node = max(node.children, key=lambda n: n.UCB1())
        board = node._get_board(fen)
        #print("\n")
        #print(board)
        #print("\n")
        if not board.is_game_over():
            node._expand(fen)
        return node

    def _expand(self, fen):
        board = self._get_board(fen)
        possible_moves = list(board.legal_moves)
        for move in possible_moves:
            self.add_child(move)

    def _get_board(self, fen):
        move_list = []
        node = self
        while node.board != None:
            move_list.append(node.board)
            node = node.parent
        board = chess.Board(fen)
        if move_list:
            for i in reversed(move_list):
                board.push(i)

        return board

    def _simulate(self, fen):
        counter = 0
        board = self._get_board(fen)

        while not board.is_game_over() and (counter <= 20):
            p_piece, p_move, val = self.nn_predictor(torch.Tensor(getBoard.get_bit_board(board)), torch.Tensor(getBoard.get_info_board(board)))
            p_piece = torch.softmax(p_piece, dim=1)
            p_move = torch.softmax(p_move, dim=1)
            move = self.sample_move_from_policy(self.calculate_policy_map(p_piece, p_move, list(board.legal_moves), board))
            board.push(move)
            counter += 1
        if (counter > 20):
            result = eval.eval(board)


        elif board.outcome().winner == None:
            result = 0.5
        else:
            if board.outcome().winner:
                result = 1
            else:
                result = 0
            if counter%2==0:
                result = 1 - result

        for _ in range(counter):
            board.pop()

        return result

    def _backpropagate(self, node, result):
        while node:
            node.update(result)
            result = 1 - result
            node = node.parent

    def get_move_eval(self):
        return self.board, self.score/self.visits

    def set_nn_predictor(self, nn_predictor):
        """
        This function sets the neural network's prediction function.
        """
        self.nn_predictor = nn_predictor

    def calculate_policy_map(self, P_piece, P_move_type, legal_moves, board):
        policy_map = {}
        board = board
        # Loop through each legal move
        for move in legal_moves:
            piece = board.piece_at(move.from_square).piece_type  # Extract the piece involved in the move
            move_type = chess_utils.get_move_type(board,move)  # Extract the type of move (move, capture, attack, etc.)

            # Multiply the probabilities to get the move probability
            total_prob = P_piece[0, piece-1] * P_move_type[0, move_type]

            # Store the move and its probability
            policy_map[move] = total_prob

        # Normalize probabilities
        total_sum = sum(policy_map.values())
        for move in policy_map:
            policy_map[move] /= total_sum

        return policy_map




    def sample_move_from_policy(self,policy_map):
        # Extract the list of moves and their corresponding probabilities
        moves = list(policy_map.keys())
        probabilities = list(policy_map.values())

        # Sample a move based on the probabilities
        chosen_move = random.choices(moves, probabilities, k=1)[0]

        return chosen_move
def run_and_save_mcts():
    try:
        with open('mcts_tree.pkl', 'rb') as file:
            node = pickle.load(file)
    except:
        node = Node(None)
    node.search(num_simulations=1000, fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    j = 0
    for i in node.children:
        print(i.board)
        j += 1
        print(j)
        print('\n')
    print(node.score)
    # Save the best node (or tree) using pickle
    with open('mcts_tree.pkl', 'wb') as f:
        pickle.dump(node, f)

    print("MCTS tree saved successfully.")


if __name__ == '__main__':
    board = chess.Board(fen='rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4')
    nn_predictor = ChessEvaluator()
    nn_predictor.load_state_dict(torch.load('current.pth'))
    node = Node(board, nn_predictor)
    p_piece, p_move, val = nn_predictor(torch.Tensor(getBoard.get_bit_board(board)),
                                             torch.Tensor(getBoard.get_info_board(board)))
    p_piece = torch.softmax(p_piece, dim=1)
    print(p_piece)
    p_move = torch.softmax(p_move, dim=1)
    print(p_move)
    pol = node.calculate_policy_map(p_piece, p_move, list(board.legal_moves), board)
    print(pol)
    print(pol.get(chess.Move.from_uci('f3e5')))
    move = node.sample_move_from_policy(pol)
    print(move)


