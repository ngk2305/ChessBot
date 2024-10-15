import math
import random
import eval
import chess
import getBoard
import torch
import time
from MCNetMini import ChessEvaluator

class Node:
    def __init__(self, board, nn_pred, parent=None):
        self.board = board  # Current state of the game
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.score = 0  # Number of wins from this node
        self.nn_predictor = nn_pred
        self.searching = True

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

    def set_searching(self,value):
        self.searching = value

    def search(self, num_simulations,time_limit_seconds,fen,continual_update):

        best_move = None
        start_time = time.time()
        self.set_searching(True)
        simu_count = 0
        last_print_time = start_time

        while self.searching and simu_count < num_simulations:
            node = self._select(self, fen)
            result = node._simulate(fen)
            self._backpropagate(node, result)
            best_move = max(
                (child for child in self.children if child.visits > 0),
                key=lambda child: child.score / child.visits,
                default=None  # Provide a fallback value if no children have visits > 0
            )
            simu_count += 1

            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time >= time_limit_seconds:
                self.searching = False  # Stop searching after time limit
            if continual_update:
                if time.time() - last_print_time >= 3:
                    print(f"Best move after {simu_count} simulations: {best_move.board}")
                    last_print_time = time.time()  # Update the last print time

        return best_move

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
            main_board = torch.tensor(getBoard.get_bit_board(board),device=torch.device('cuda'))
            xtra_info = torch.tensor(getBoard.get_info_board(board), device=torch.device('cuda'))
            p_from, p_to = self.nn_predictor(main_board, xtra_info)
            p_from = torch.softmax(p_from, dim=1)
            p_to = torch.softmax(p_to, dim=1)
            move = self.sample_move_from_policy(self.calculate_policy_map(p_from, p_to, list(board.legal_moves)))
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

    def calculate_policy_map(self, P_from, P_to, legal_moves):
        policy_map = {}
        # Loop through each legal move
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square

            # Multiply the probabilities to get the move probability
            total_prob = P_from[0, from_sq] * P_to[0, to_sq]

            # Store the move and its probability
            policy_map[move] = total_prob

        # Normalize probabilities
        total_sum = sum(policy_map.values())
        epsilon = 0.05
        for move in policy_map:
            policy_map[move] = ((policy_map[move]/total_sum) + epsilon) / (1 +epsilon*len(policy_map))
        policy_map = dict(sorted(policy_map.items(), key=lambda item: item[1], reverse=True))
        return policy_map




    def sample_move_from_policy(self,policy_map):
        # Extract the list of moves and their corresponding probabilities
        moves = list(policy_map.keys())
        probabilities = list(policy_map.values())

        # Sample a move based on the probabilities
        chosen_move = random.choices(moves, probabilities, k=1)[0]

        return chosen_move


if __name__ == '__main__':
    board = chess.Board(fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    nn_predictor = ChessEvaluator()
    nn_predictor.eval()
    try:
        nn_predictor.load_state_dict(torch.load('epoch13.pth'))
    except:
        print('cant load')
    node = Node(board, nn_predictor)
    p_piece, p_move = nn_predictor(torch.Tensor(getBoard.get_bit_board(board)),
                                             torch.Tensor(getBoard.get_info_board(board)))

    p_piece = torch.softmax(p_piece, dim=1)
    print(p_piece)
    p_move = torch.softmax(p_move, dim=1)
    print(p_move)
    pol = node.calculate_policy_map(p_piece, p_move, list(board.legal_moves))
    print(pol)
    move = node.sample_move_from_policy(pol)
    print(move)


