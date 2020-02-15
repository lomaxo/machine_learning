import random
import numpy as np
import neural_net
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
        for i in range(self.GRID_SIZE):
            if self.grid_list[i + i*self.GRID_SIZE] != marker:
                match = False
        if match: return True
        for i in range(self.GRID_SIZE):
            if self.grid_list[(self.GRID_SIZE-i-1) + i*self.GRID_SIZE] != marker:
                match = False
        if match: return True

    def get_best_move(self, net, player):
        options = self.get_empty()
        if len(options) == 0:
            return None
        for move in options:
            new_grid = copy(self)
            new_grid.set_pos(player)
            in_matrix = game_state_to_matrix(new_grid.grid_list)
            out = net.feedforward(in_matrix)
            print(out)

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
            #grid.display_grid()
            positions.append(list(grid.grid_list))

        if not draw:
            #print(f"Winner is {player_marker}")
            #print(positions)
            for pos in positions:
                training_data.append((pos, player_marker))
        else:
            #print("Draw")
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

def test_game():
    for state, winner in formatted_training_data:
        out = net.feedforward(state)
        score = out[0][0]*(-1) + out[1][0]
        print(f"{out.T}, {winner.T}, {score}")

def play_game(net):
    grid = TTTGrid()
    player_marker = 'x'
    draw = False
    while not grid.vertical_test(player_marker) and not grid.horizontal_test(player_marker) and not grid.diagonal_test(player_marker) and not draw:
        if player_marker == 'x':
            player_marker = 'o'
        else:
            player_marker = 'x'
        next_mark = grid.get_best_move(net, player_marker)
        if not next_mark:
            draw = True
        else:
            grid.set_pos(next_mark, player_marker)
            grid.display_grid()

    if not draw:
        print(f"Winner is {player_marker}")
    else:
        print("Draw")

if __name__ == "__main__":
    random_games = get_random_training_data(100)
    random.shuffle(random_games)
    print("Size of training data:", len(random_games))
    formatted_training_data = format_training_data(random_games)

    # Create a new net or load the previous one.
    # net = neural_net.Network([18, 20, 20, 2])
    net = neural_net.load_net('net.p')

    # Train
    print("Training...")
    for _ in range(100):
        net.update_from_batch(formatted_training_data, 1.0)
    net.save_net('net.p')
    test_game()
