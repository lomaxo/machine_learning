import random
import numpy as np
class TTTGrid:
    def __init__(self):
        self.GRID_SIZE = 3
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





training_data = []
for _ in range(10):
    grid = TTTGrid()
    player_marker = 'x'
    draw = False
    positions = []
    while not grid.vertical_test(player_marker) and not grid.horizontal_test(player_marker) and not draw:
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

print(training_data)
formatted_training_data = []
for item, winner in training_data:
    input_array = []
    for i in item:
        if i == 'x':
            input_array.extend([1,0])
        elif i == 'o':
            input_array.extend([0,1])
        else:
            input_array.extend([0,0])
    if winner == 'x':
        win = [1,0]
    elif winner == 'o':
        win = [0,1]
    else:
        win = [0,0]
    formatted_training_data.append((np.array(input_array), win))
print(formatted_training_data)
