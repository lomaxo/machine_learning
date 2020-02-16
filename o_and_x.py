import random
import numpy as np
import neural_net
from copy import deepcopy
import pickle

class TTTGrid:
    def __init__(self, grid_list = []):
        self.GRID_SIZE = 3
        self.current_player = 'x'
        if len(grid_list) > 0 and len(grid_list) != self.GRID_SIZE**2:
            raise("Wrong grid size")

        if len(grid_list) > 0:
            self.grid_list = grid_list
        else:
            self.grid_list = [None] * self.GRID_SIZE**2

    def next_player(self):
        if self.current_player == 'x':
            self.current_player = 'o'
        else:
            self.current_player = 'x'

    def get_empty(self):
        ret = []
        for i, pos in enumerate(self.grid_list):
            if pos == None:
                ret.append(i)
        return ret

    def set_pos(self, pos, marker = None):
        if not marker:
            marker = self.current_player
        self.grid_list[pos] = marker

    def display_grid(self):
        for row_i in range(0, len(self.grid_list), self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                    mark = self.grid_list[row_i + i]
                    if not mark: mark = '~'
                    print(mark, end='')
            print("\n")
        print("------------")

    def horizontal_test(self, marker = None):
        if not marker:
            marker = self.current_player
        for row_i in range(0, len(self.grid_list), self.GRID_SIZE):
            match = True
            for i in range(self.GRID_SIZE):
                if self.grid_list[row_i + i] != marker:
                    match = False
            if match: return True

    def vertical_test(self, marker = None):
        if not marker:
            marker = self.current_player
        for column_i in range(0, self.GRID_SIZE):
            match = True
            for i in range(self.GRID_SIZE):
                if self.grid_list[column_i + i*self.GRID_SIZE] != marker:
                    match = False
            if match: return True

    def diagonal_test(self, marker = None):
        if not marker:
            marker = self.current_player
        match = True
        if self.grid_list[0] == marker and self.grid_list[4] == marker and self.grid_list[8] == marker: return True
        if self.grid_list[2] == marker and self.grid_list[4] == marker and self.grid_list[6] == marker: return True
        return False
        # for i in range(self.GRID_SIZE):
        #     if self.grid_list[i + i*self.GRID_SIZE] != marker:
        #         match = False
        # if match: return True
        # for i in range(self.GRID_SIZE):
        #     if self.grid_list[(self.GRID_SIZE-i-1) + i*self.GRID_SIZE] != marker:
        #         match = False
        # if match: return True

    def test_win(self, marker = None):
        if not marker:
            marker = self.current_player
        return self.vertical_test(marker) or self.horizontal_test(marker) or self.diagonal_test(marker)

    def get_best_move(self, net, player = None):
        if not player:
            player = self.current_player
        options = self.get_empty()
        if len(options) == 0:
            return None
        results = []
        for move in options:
            new_grid = deepcopy(self)
            new_grid.set_pos(move, player)
            score = get_score(net, new_grid.grid_list)
            results.append((move, score))
        if player == 'x':
            best_move = min(results, key=lambda item: item[1])[0]
        else:
            best_move = max(results, key=lambda item: item[1])[0]
        # print(best_move)
        return best_move

    def get_random_move(self, net, player = None):
        if not player:
            player = self.current_player
        options = self.get_empty()
        if len(options) == 0:
            return None
        return random.choice(options)

def get_matrix_from_marker(marker):
    if marker == 'x':
        return np.array([[1,0]]).transpose()
    elif marker == 'o':
        return np.array([[0,1]]).transpose()
    else:
        return np.array([[0,0]]).transpose()


def game_state_to_matrix(grid_list):
    ret_array = []
    for i in grid_list:
        if i == 'x':
            ret_array.extend([1,0])
        elif i == 'o':
            ret_array.extend([0,1])
        else:
            ret_array.extend([0,0])
    return np.array([ret_array]).transpose()

# TODO: Tidy this function to use grid.test_win() and grid.next_player
def get_random_training_data(n_games):
    # Create training data
    training_data = []
    for _ in range(n_games):
        grid = TTTGrid()
        player_marker = 'x'
        draw = False
        positions = []
        while not grid.vertical_test(player_marker) and not grid.horizontal_test(player_marker) and not grid.diagonal_test(player_marker) and not draw:
            if player_marker == 'x':
                player_marker = 'o'
            else:
                player_marker = 'x'
            options = grid.get_empty()
            if options:
                next_mark = random.choice(options)
            else:
                draw = True
            grid.set_pos(next_mark, player_marker)
            positions.append(list(grid.grid_list))

        if not draw:
            for pos in positions:
                training_data.append((pos, player_marker))
        else:
            for pos in positions:
                training_data.append((pos, '-'))
    return training_data



def format_training_data(game_data):
    formatted_training_data = []
    for game_state, winner in game_data:
        nn_input = game_state_to_matrix(game_state)
        win = get_matrix_from_marker(winner)
        formatted_training_data.append((nn_input, win))
    return formatted_training_data

def get_score(net, grid_list):
    in_matrix = game_state_to_matrix(grid_list)
    out = net.feedforward(in_matrix)
    return out[0][0]*(-1) + out[1][0]


def test_game():
    for state, winner in formatted_training_data:
        out = net.feedforward(state)
        score = out[0][0]*(-1) + out[1][0]
        print(f"{out.T}, {winner.T}, {score}")

def play_game(net, grid, o_move_func, x_move_func, display=False):
    # grid = TTTGrid()
    #player_marker = 'x'
    draw = False
    positions = []
    while not grid.test_win() and not draw:
        grid.next_player()
        if grid.current_player == 'x':
            next_mark = x_move_func(net)
        else:
            next_mark = o_move_func(net)
        if next_mark == None:
            draw = True
        else:
            grid.set_pos(next_mark)
            if display: grid.display_grid()
        positions.append(grid.grid_list)
    if not draw:
        if display: print(f"Winner is {grid.current_player}")
        winner = grid.current_player
    else:
        if display: print("Draw")
        winner = '-'
    data = []
    for pos in positions:
        data.append((pos, winner))
    return data

def test_multiple_games(net, n_games, start_player, o_random = False, x_random = False):
    wins = [0,0,0]
    for g in range(n_games):
        grid = TTTGrid()
        if x_random: x_func = grid.get_random_move
        else: x_func = grid.get_best_move

        if o_random: o_func = grid.get_random_move
        else: o_func = grid.get_best_move

        grid.current_player = start_player
        grid.next_player()
        winner = play_game(net, grid, o_func, x_func, False)[-1][1]
        if winner == 'o':
            wins[0] += 1
        elif winner == 'x':
            wins[1] += 1
        else:
            wins[2] += 1
    return wins

if __name__ == "__main__":
    # Create some random test data

    random_games = get_random_training_data(10000)
    random.shuffle(random_games)
    print("Size of training data:", len(random_games))
    formatted_training_data = format_training_data(random_games)

    # Create a new net or load the previous one.

    # net = neural_net.Network([18, 20, 20, 2])
    net = neural_net.load_net('net.p')

    # Train
    # print("Training.")
    # print("Random training data...")
    # net.SGD(formatted_training_data, 1, 100, .5)
    # for _ in range(100):
    #     net.update_from_batch(formatted_training_data, 1.0)

    # print("Self-play...")
    # for _ in range(100):
    #     data = play_game(net)
    #     net.update_from_batch(format_training_data(data), 1.0)
    # net.save_net('net.p')

    # Play some games to test how successful it is
    print("\no: NN, x: NN, starter: o")
    print(test_multiple_games(net, 100, 'o'))
    print("\no: NN, x: NN, starter: x")
    print(test_multiple_games(net, 100, 'x'))

    print("\no: NN, x: random, starter: o")
    print(test_multiple_games(net, 100, 'o', False, True))
    print("\no: NN, x: random, starter: x")
    print(test_multiple_games(net, 100, 'x', False, True))

    print("\no: random, x: NN, starter: o")
    print(test_multiple_games(net, 100, 'o', True, False))
    print("\no: random, x: NN, starter: x")
    print(test_multiple_games(net, 100, 'x', True, False))
