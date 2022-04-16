# -*- coding: utf-8 -*-

def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def readInput2(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def writeOutput(result, path='output.txt'):
    res = ""
    if result == "PASS":
    	res = "PASS"
    else:
	    res += str(result[0]) + ',' + str(result[1])
    with open(path, 'w') as f:
        f.write(res)



"""# Game State Function

"""

import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy

class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def encode_state(self,s):
        """ Encode the current state of the board as a string
        """
        state = []
        state = s
        return ''.join([str(state[i][j]) for i in range(0,5) for j in range(0,5)])

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
        self.opponent = 1 if self.piece_type == 2 else 2

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''

        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_board(self, board,i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self,i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def detect_neighbor_ally_board(self,board,i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        neighbors = self.detect_neighbor_board(board,i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies
    
    def detect_neighbor_ally_if_place(self,board,i, j ,piece_type):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        neighbors = self.detect_neighbor_board(board,i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == piece_type:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self , i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_liberty_board(self,board, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor_board(board,member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces
    
    def find_died_pieces_board(self,board,piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty_board(board,i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces    

    def remove_died_pieces_board(self,board ,piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''
        died_pieces = self.find_died_pieces_board(board,piece_type)
        if not died_pieces: return board
        new_board = self.remove_certain_pieces_board(board,died_pieces)
        return new_board


    def remove_certain_pieces_board(self, board, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        new_board = board
        return new_board

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def next_state_after_move(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        state_move_dict = {}
        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return None, None
        next_board = deepcopy(self.board)
        next_board[i][j] = piece_type

        next_board = self.remove_died_pieces_board(next_board,self.opponent)
        return next_board, (i,j)        

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False
        
        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

       # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def get_legal_states(self):
        if self.game_end(self.piece_type):
          return []
        legal_dict = {}
        legal_state = []
        legal_move = []
        for row in range(0,5):
          for col in range(0,5):
            state, move = self.next_state_after_move(row, col,self.piece_type)
            #move = self.encode_state(move)
            if move is not None:
              legal_state.append(state)
              legal_move.append(move)
        return legal_state,legal_move

    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt          

    def end_score(self):
        cnt_1 = self.score(1) + self.komi
        cnt_2 = self.score(2) + self.komi
        return cnt_1, cnt_2   

    def opponent(self):
        return 1  if self.piece_type == 2 else 1

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--move", "-m", type=int, help="number of total moves", default=0)
#     parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
#     args = parser.parse_args()

#     judge(args.move, args.verbose)



"""

```
# This is formatted as code
```

# Alpha Beta Agent"""

import random
from queue import PriorityQueue

MAX_SCORE = 999999
MIN_SCORE = -999999

def alpha_beta_result(state, max_depth, best_black, best_white, eval_fn):
    # if game_state.is_over():                                   # <1>
    #     if game_state.winner() == game_state.next_player:      # <1>
    #         return MAX_SCORE                                   # <1>
    #     else:                                                  # <1>
    #         return MIN_SCORE                                   # <1>

    if max_depth == 0:                                         # <2>
        #print(state.board)
        #print(eval_fn(state))
        #print('Value return: ',eval_fn(state))
        return eval_fn(state)                             # <2>

    best_so_far = MIN_SCORE
    #print(state.get_legal_states())
    possible_state_list = state.get_legal_states()[0]
    possible_move_list = state.get_legal_states()[1]
    #print(possible_state_list)
    #for possible_move in state_move_dict:
    for i in range (0,len(possible_state_list)):
        possible_state = possible_state_list[i]
        possible_move = possible_move_list[i]
        next_state = GO(5)
        next_state.set_board(state.opponent, state.board, possible_state)   # <4>
        opponent_best_result = alpha_beta_result(              # <5>
            next_state, max_depth - 1,                         # <5>
            best_black, best_white,                            # <5>
            eval_fn)                                           # <5>
        our_result = -1 * opponent_best_result                 # <6>
        #print('our_result: ', our_result)
        #print('best_so_far: ', best_so_far)
        if our_result > best_so_far:                           # <7>
            best_so_far = our_result                           # <7>
# end::alpha-beta-prune-1[]

# tag::alpha-beta-prune-2[]
        if state.piece_type == 2:
            if best_so_far > best_white:                       # <8>
                best_white = best_so_far                       # <8>
            outcome_for_black = -1 * best_so_far               # <9>
            if outcome_for_black < best_black:                 # <9>
                return best_so_far                             # <9>
# end::alpha-beta-prune-2[]
# tag::alpha-beta-prune-3[]
        elif state.piece_type == 1:
            if best_so_far > best_black:                       # <10>
                best_black = best_so_far                       # <10>
            outcome_for_white = -1 * best_so_far               # <11>
            if outcome_for_white < best_white:                 # <11>
                return best_so_far                             # <11>
# end::alpha-beta-prune-3[]
# tag::alpha-beta-prune-4[]
    #print('Best so far: ',best_so_far)
    return best_so_far

class AlphaBetaAgent():
    def __init__(self, max_depth, eval_fn):
        self.max_depth = max_depth
        self.eval_fn = eval_fn
        self.type = 'Alpha Beta'
        self.cnt = 1

    def detect_neighbor_board(self, board,i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors    
        
    def number_of_liberty(self,state,move):
      i,j = move[0], move[1]
      neighbors = self.detect_neighbor_board(state,i, j)
      liberty = 0
      for n in neighbors:
        if state[n[0]][n[1]] == 0:
          liberty += 1
      return liberty

    def count_my_piece(self, state):
      board = state.board
      black_stones = 0
      white_stones = 0
      for r in range(0,5):
        for c in range(0,5):
            if board[r][c] == 1:
                black_stones += 1
            elif board[r][c] == 2:
                white_stones += 1
    #print(board)
    #print('black stone: ',black_stones)
    #print('white stone: ', white_stones)
    #print('difference', diff)
      if state.piece_type == 1:    # <2>
        return black_stones                                       # <2>
      else:
        return white_stones 
    
    def count_edge_move(self, moves):
      edges = [(0,0),(4,0),(0,4),(4,4)]
      number_of_edge_move = 0
      for m in moves:
        if m in edges:
          number_of_edge_move += 1
      return number_of_edge_move

    def remove_edge_move(self, moves):
      edges = [(0,0),(4,0),(0,4),(4,4)]
      if (0,0) in moves:
        moves.remove((0,0))
      if (4,0) in moves:
        moves.remove((4,0))
      if (0,4) in moves:
        moves.remove((0,4))
      if (4,4) in moves:
        moves.remove((4,4))
      return moves

    def get_input(self, go ,piece_type): ## select_move
        best_moves_list = []
        best_moves_l = []
        best_moves = PriorityQueue()
        most_liberty = 0
        best_score = -1000
        best_black = MIN_SCORE
        best_white = MIN_SCORE
        # Loop over all legal moves.
        #print(go.get_legal_states())
        if(self.count_my_piece(go) == 6 ):
         # print('Depth Adjusted to 3')
          self.max_depth = 2
        if(self.count_my_piece(go) == 7 ):
         # print('Depth Adjusted to 2')
          self.max_depth = 3
        if(self.count_my_piece(go) == 8):
         # print('Depth Adjusted to 2')
          self.max_depth = 2
        if(self.count_my_piece(go) == 9):
         # print('Depth Adjusted to 1')
          self.max_depth = 2
        if(self.count_my_piece(go) >= 10):
         # print('Depth Adjusted to 1')
          self.max_depth = 1
        if go.board == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]:
          self.cnt += 1
          return (2,2)
        if go.piece_type == 2 and self.count_my_piece(go) == 0 and go.board[2][2] == 0:
          return (2,2)
        if go.piece_type == 2 and self.count_my_piece(go) == 0 and go.board[2][2] != 0:
          return (2,1)
        if self.count_my_piece(go) <= 3 and go.board[2][2] != 0:
          opening = []
          if go.board[1][2] == 0:
              opening.append((1,2))
          if go.board[3][2] == 0:
              opening.append((3,2))
          if go.board[2][3] == 0:
              opening.append((2,3))
          if go.board[2][1] == 0:
              opening.append((2,1))
          if len(opening) == 0:
              if go.board[2][1] == go.piece_type and go.board[1][0] == 0 :
                  opening.append((1,0))
              if go.board[2][1] == go.piece_type and go.board[3][0] == 0 :
                  opening.append((3,0))
              if go.board[1][2] == go.piece_type and go.board[0][1] == 0 :
                  opening.append((0,1))
              if go.board[1][2] == go.piece_type and go.board[0][3] == 0 :
                  opening.append((0,3))
              if go.board[2][3] == go.piece_type and go.board[1][4] == 0 :
                  opening.append((1,4))
              if go.board[2][3] == go.piece_type and go.board[3][4] == 0 :
                  opening.append((3,4))
              if go.board[3][2] == go.piece_type and go.board[4][1] == 0 :
                  opening.append((4,1))
              if go.board[3][2] == go.piece_type and go.board[4][3] == 0 :
                  opening.append((4,3))
              return random.choice(opening) 
          if len(opening) != 0:
              return random.choice(opening)
        possible_state_list = go.get_legal_states()[0]
        possible_move_list = go.get_legal_states()[1]
        #for possible_move in state_move_dict:
        for i in range (0,len(possible_state_list)):
            possible_state = possible_state_list[i]
            possible_move = possible_move_list[i]
            #move = possible_move
            #possible_state = go.get_legal_states()[possible_move]
            #print(go.opponent, go.board, possible_move)
            # Calculate the game state if we select this move.
            next_state = GO(5)
            #print(possible_state)
            next_state.set_board(go.opponent, go.board, possible_state)
        #     # Since our opponent plays next, figure out their best
        #     # possible outcome from there.\
            #print('current board',go.board)
            #print('current move', possible_move)
            #print('next state board', next_state.board)
            opponent_best_outcome = alpha_beta_result(next_state, self.max_depth,best_black, best_white,self.eval_fn) 
        #     # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best_outcome
            #print('our best outcome', our_best_outcome)
           # print('best score ', best_score)
            if (not best_moves) or our_best_outcome > best_score:
        #         # This is the best move so far.
                #best_moves.append(possible_move)
                #print('to be appended:', possible_move)
                #print(our_best_outcome)
                best_moves.put((-our_best_outcome,possible_move))
                best_score = our_best_outcome
                if go.piece_type == 1:
                    best_black = best_score
                elif go.piece_type == 2:
                    best_white = best_score
            elif our_best_outcome == best_score:
                # This is as good as our previous best move.
               # best_moves.append(possible_move)
               #print('to be appended:', possible_move)
               #print(our_best_outcome)
               best_moves.put((-our_best_outcome,possible_move))
        if best_moves.qsize() == 0:
          return "PASS"
          
        #print(our_best_outcome)
        bestcost, bestmove = best_moves.get()
        best_moves_list.append(bestmove)
        #print(best_moves.qsize())
        for i in range (0,best_moves.qsize()):
          cost, move = best_moves.get()
          if cost == bestcost:
            best_moves_list.append(move)  
        #print(best_moves_list)
        if  len(best_moves_list) ==0:
          return "PASS"

        # check number of edge moves and try to avoid them      
        num_edge_move = self.count_edge_move(best_moves_list)
        if(len(best_moves_list) > num_edge_move):
          best_moves_list = self.remove_edge_move(best_moves_list)    
        ############################################################
        
        
         #### prioritize max liberty move #####
        for i in range(0,len(best_moves_list)):
          n_liberty = self.number_of_liberty(go.board,best_moves_list[i])
          if most_liberty == 0:
            most_liberty = n_liberty
            best_moves_l.append(best_moves_list[i])
          elif n_liberty > most_liberty:
            best_moves_l.clear()
            best_moves_l.append(best_moves_list[i])
          elif n_liberty == most_liberty:
            best_moves_l.append(best_moves_list[i])
        
        return random.choice(best_moves_l)
            
        #best_moves_max_neighbor = []
        #most_neighbor = 0
        #for i in range(0,len(best_moves_l)):
          #r,c = best_moves_l[i]
         # number_of_neighbor = len(go.detect_neighbor_ally_if_place(go.board,r,c,go.piece_type))
          #if number_of_neighbor > most_neighbor and number_of_neighbor <3 and self.count_my_piece(go) < 3:
           # most_neighbor = number_of_neighbor
          #  best_moves_max_neighbor.clear()
          #  best_moves_max_neighbor.append(best_moves_l[i])
         # number_of_neighbor = 0
        # For variety, randomly select among all equally good moves.
        #self.cnt += 1
        #if len(best_moves_max_neighbor) == 0:
         # return random.choice(best_moves_l)
        #else:
         # return random.choice(best_moves_max_neighbor)  # return random from the most neighbor chess
# end::alpha-beta-agent[]

"""# Main"""

def capture_diff(state):
    board = state.board
    black_stones = 0
    white_stones = 0
    for r in range(0,5):
        for c in range(0,5):
            if board[r][c] == 1:
                black_stones += 1
            elif board[r][c] == 2:
                white_stones += 1
    diff = black_stones - white_stones              # <1>
    #print(board)
    #print('black stone: ',black_stones)
    #print('white stone: ', white_stones)
    #print('difference', diff)
    if state.piece_type == 1:    # <2>
        return diff                                       # <2>
    return -1 * diff                                      # <3>


if __name__ == '__main__':
  N = 5
  piece_type, previous_board, board = readInput(N)
  #print(piece_type, previous_board, board)
  #print(capture_diff(go.board))
  #legal_states = go.get_legal_states()
  #print(legal_states[0])
  player1 = AlphaBetaAgent(2, capture_diff)
  go = GO(N)
  go.set_board(piece_type, previous_board, board)
  move = player1.get_input(go,piece_type)
  print(move)
  writeOutput(move)