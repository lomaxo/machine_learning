import random
import numpy as np
import neural_net
from copy import deepcopy

class TTTGrid:
    def __init__(self, start_player = 'o', grid_list = []):
        self.GRID_SIZE = 3
        self.current_player = start_player
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

    def get_random_move(self, player = None):
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

# TODO: Tidy this function to use grid.test_win() and grid.next_player. OR remove because we can use play_game to produce random test dat now.
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



def format_game_data(game_data):
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
        out = net_v1.feedforward(state)
        score = out[0][0]*(-1) + out[1][0]
        print(f"{out.T}, {winner.T}, {score}")

def play_game(grid, o_net, x_net, display=False):
    draw = False
    positions = []
    grid.next_player()
    while not grid.test_win() and not draw:
        grid.next_player()
        if grid.current_player == 'x':
            if x_net == None:
                next_mark = grid.get_random_move()
            else:
                next_mark = grid.get_best_move(x_net)
        else:
            if o_net == None:
                next_mark = grid.get_random_move()
            else:
                next_mark = grid.get_best_move(o_net)
        if next_mark == None:
            draw = True
        else:
            grid.set_pos(next_mark)
            if display: grid.display_grid()
        positions.append(deepcopy(grid.grid_list))
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

def test_multiple_games(n_games, start_player, o_net, x_net):
    wins = {'o': 0,'x': 0, 'draw': 0}
    for g in range(n_games):
        winner = play_game(TTTGrid(start_player), o_net, x_net, False)[-1][1]
        if winner == 'o':
            wins['o'] += 1
        elif winner == 'x':
            wins['x'] += 1
        else:
            wins['draw'] += 1
    return wins

def compare(net1, net2, nn1 = 'NN1', nn2 = 'NN2'):
    # Play some games to test how successful it is
    if net1: name1 = net1.name
    else: name1 = 'Random'
    if net2: name2 = net2.name
    else: name2 = 'Random'
    print(f"o: {name1}, x: {name2}, starter: o  ->  ", end='')
    print(test_multiple_games(100, 'o', net1, net2))

    print(f"o: {name2}, x: {name1}, starter: o  ->  ", end='')
    print(test_multiple_games(100, 'o', net2, net1))

    # Can also compare when x plays first
    # print(f"o: {nn1}, x: {nn2}, starter: x  --  ", end='')
    # print(test_multiple_games(100, 'x', net1, net2))
    # print(f"o: {nn2}, x: {nn1}, starter: x  --  ", end='')
    # print(test_multiple_games(100, 'x', net2, net1))

def train(master_net, training_net):
    print(f"\nCreating training data...")
    data = []
    for g in range(1000):
        # Play against previous net
        data.extend(play_game(TTTGrid('o'), master_net, training_net, False))
        data.extend(play_game(TTTGrid('o'), training_net, master_net, False))
        # Play against random
        data.extend(play_game(TTTGrid('o'), None, training_net, False))
        data.extend(play_game(TTTGrid('o'), training_net, None, False))

        # Train with x going first
        # data.extend(play_game(TTTGrid('x'), net_v1, net_latest, False))
        # data.extend(play_game(TTTGrid('x'), net_latest, net_v1, False))
        # data.extend(play_game(TTTGrid('x'), None, net_latest, False))
        # data.extend(play_game(TTTGrid('x'), net_latest, None, False))

    print("SGD Training...")
    net_latest.SGD(format_game_data(data), 10, 1000, 0.5)


if __name__ == "__main__":
    # Create a new net or load the previous one.
    net_v1 = neural_net.load_net('net.p')
    net_max = neural_net.load_net('max.p')
    working_net_name = "wilma"
    working_net_filename = working_net_name + '.p'
    try:
        net_latest = neural_net.load_net(working_net_filename)
    except (OSError, IOError):
        print(f"Creating new network {working_net_filename}.")
        net_latest = neural_net.Network([18, 20, 20, 2], working_net_name)


    compare(net_v1, None)
    compare(net_max, None)
    compare(net_v1, net_max)

    # Train
    score_as_o = []
    score_as_x = []
    # while True:
    #     score_as_o.append(test_multiple_games(100, 'o', net_latest, None))
    #     score_as_x.append(test_multiple_games(100, 'o', None, net_latest))
    #     print(f"Play first: {score_as_o[-1]['o']}% win, Play second: {score_as_x[-1]['x']}% win")
    #     with open(net_latest.name+'_score.csv', 'a') as score_file:
    #         score_file.write(f"{score_as_o[-1]['o']}, {score_as_x[-1]['x']}\n")
    #     train(None, net_latest)
    #     net_latest.save_net()
    # Display a game of the latest net playing against itself to see how well it plays.
    # play_game(TTTGrid('o'), net_v1, None, True)
    # play_game(TTTGrid('o'), net_max, None, True)
