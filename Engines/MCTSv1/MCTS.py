import math
import random
import pickle
import eval
import chess
from tqdm import tqdm

# Step 1: Create a Node class
class Node:
    def __init__(self, board ,parent=None):
        self.board = board  # Current state of the game
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.score = 0  # Number of wins from this node

    def add_child(self, child_state):
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.score += result

    def UCB1(self, exploration_param=0.125):
        if self.visits == 0:
            return math.inf  # Assign infinite value if never visited
        return (self.score / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def search(self, num_simulations):
        for _ in tqdm(range(num_simulations)):
            node = self._select(self)
            result = self._simulate()
            self._backpropagate(node, result)


        return max(self.children, key=lambda child: child.visits)  # Best child after simulations

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: n.UCB1())
        board = self._get_board()
        if not board.is_game_over():
            node._expand()
        return node

    def _expand(self):
        board = self._get_board()
        possible_moves = list(board.legal_moves)
        for move in possible_moves:
            self.add_child(move)

    def _get_board(self):
        move_list=[]
        node = self
        while node.board != None:
            move_list.append(node.board)
            node = node.parent
        board = chess.Board()
        if move_list:
            for i in reversed(move_list):
                board.push(i)

        return board




    def _simulate(self):
        counter = 0
        board = self._get_board()
        while not board.is_game_over() and (counter <= 19):
            move = random.choice(list(board.legal_moves))
            board.push(move)
            counter += 1
        if (counter > 19):
            result = eval.eval(board)
        elif board.outcome().winner:
            result = 1
        elif board.outcome().winner == None:
            result = 0.5
        else:
            result = 0
        for _ in range(counter):
            board.pop()
        return result

    def _backpropagate(self, node, result):
        while node:
            node.update(result)
            result = 1 - result
            node = node.parent





def run_and_save_mcts():
    try:
        with open('mcts_tree.pkl', 'rb') as file:
            node = pickle.load(file)
    except:
        node = Node(None)
    node.search(num_simulations=100000)
    j=0
    for i in node.children:
        print(i.board)
        j+=1
        print(j)
        print('\n')
    print(node.score)
    # Save the best node (or tree) using pickle
    with open('mcts_tree.pkl', 'wb') as f:
        pickle.dump(node, f)

    print("MCTS tree saved successfully.")


run_and_save_mcts()