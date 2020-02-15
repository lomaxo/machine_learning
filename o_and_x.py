import random
import numpy as np
import neural_net
from copy import deepcopy
import pickle

class TTTGrid:
    def __init__(self, grid_list = []):
        self.GRID_SIZE = 3
        if len(grid_list) > 0 and len(grid_list) != self.GRID_SIZE**2:
            raise("Wrong grid size")

        if len(grid_list) > 0:
            self.grid_list = grid_list
        else:
            self.grid_list = [None] * self.GRID_SIZE**2

    def get_empty(self):
        ret = []
        for i, pos in enumerate(self.grid_list):
            if pos == None:
                ret.append(i)
        return ret

    def set_pos(self, pos, marker):
        self.grid_list[pos] = marker

    def display_grid(self):
        for row_i in range(0, len(self.grid_list), self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                    mark = self.grid_list[row_i + i]
                    if not mark: mark = '~'
                    print(mark, end='')
            print("\n")
        print("------------")

    def horizontal_test(self, marker):
        for row_i in range(0, len(self.grid_list), self.GRID_SIZE):
            match = True
            for i in range(self.GRID_SIZE):
                if self.grid_list[row_i + i] != marker:
                    match = False
            if match: return True

    def vertical_test(self, marker):
        for column_i in range(0, self.GRID_SIZE):
            match = True
            for i in range(self.GRID_SIZE):
                if self.grid_list[column_i + i*self.GRID_SIZE] != marker:
                    match = False
            if match: return True

    def diagonal_test(self, marker):
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

    def test_win(self, player_marker):
        return self.vertical_test(player_marker) or self.horizontal_test(player_marker) or self.diagonal_test(player_marker)

    def get_best_move(self, net, player):
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

def play_game(net, display=False):
    grid = TTTGrid()
    player_marker = 'x'
    draw = False
    positions = []
    while not grid.test_win(player_marker) and not draw:
        if player_marker == 'x':
            player_marker = 'o'
        else:
            player_marker = 'x'
        next_mark = grid.get_best_move(net, player_marker)
        if next_mark == None:
            draw = True
        else:
            grid.set_pos(next_mark, player_marker)
            if display: grid.display_grid()
        positions.append(grid.grid_list)
    if not draw:
        if display: print(f"Winner is {player_marker}")
        winner = player_marker
    else:
        if display: print("Draw")
        winner = '-'
    data = []
    for pos in positions:
        data.append((pos, winner))
    return data

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
    print("Training.")
    print("Random training data...")
    net.SGD(formatted_training_data, 10, 1000, 1)
    # for _ in range(100):
    #     net.update_from_batch(formatted_training_data, 1.0)

    # print("Self-play...")
    # for _ in range(100):
    #     data = play_game(net)
    #     net.update_from_batch(format_training_data(data), 1.0)
    net.save_net('net.p')
    play_game(net, True)
