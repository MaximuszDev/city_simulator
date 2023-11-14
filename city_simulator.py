import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class CityGrid:
    def __init__(self, rows, columns, block_percentage=30, tower_cost=1, budget_limit=100):
        self.rows = rows
        self.columns = columns
        self.grid = self.generate_grid(block_percentage)
        self.tower_cost = tower_cost
        self.budget_limit = budget_limit
        self.total_spent = 0

    def generate_grid(self, block_percentage):
        grid = np.zeros((self.rows, self.columns))

        for i in range(self.rows):
            for j in range(self.columns):
                if random.randint(1, 100) <= block_percentage:
                    grid[i][j] = 1

        return grid

    def display_grid(self):
        for row in self.grid:
            print(" ".join(map(str, row)))

    def place_tower(self, x, y, radius):
        if self.total_spent + self.tower_cost <= self.budget_limit:
            self.grid[x][y] = 3
            self.total_spent += self.tower_cost

            for i in range(max(0, x - radius), min(self.rows, x + radius + 1)):
                for j in range(max(0, y - radius), min(self.columns, y + radius + 1)):
                    if (x - i) ** 2 + (y - j) ** 2 <= radius ** 2 and self.grid[i][j] != 3:
                        self.grid[i][j] = 2

    def is_valid_position(self, x, y, radius):
        for i in range(max(0, x - radius), min(self.rows, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.columns, y + radius + 1)):
                if not (0 <= x < self.rows and 0 <= y < self.columns and self.grid[x][y] == 0 and self.grid[i][j] != 2):
                    return False
        return True

    def optimize_tower_placement(self, radius):
        free_blocks = np.argwhere(self.grid == 0)
        while len(free_blocks) > 0:
            max_coverage = 0
            best_position = None

            for position in free_blocks:
                x, y = position
                if self.is_valid_position(x, y, radius):
                    coverage = np.sum(
                        self.is_valid_position(x - 1, y, radius) +
                        self.is_valid_position(x + 1, y, radius) +
                        self.is_valid_position(x, y - 1, radius) +
                        self.is_valid_position(x, y + 1, radius)
                    )

                    if coverage > max_coverage:
                        max_coverage = coverage
                        best_position = position

            if best_position is not None:
                x, y = best_position
                self.place_tower(x, y, radius)
                free_blocks = np.argwhere(self.grid == 0)
            else:
                break

    def find_most_reliable_path(self, start, end):
        visited = set()
        queue = deque([(start, [])])

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path + [current]

            if current not in visited:
                visited.add(current)
                neighbors = self.get_neighbors(current)
                queue.extend((neighbor, path + [current]) for neighbor in neighbors)

        return None

    def get_neighbors(self, position):
        x, y = position
        neighbors = []

        if 0 <= x - 1 < self.rows and self.grid[x - 1][y] != 1:
            neighbors.append((x - 1, y))
        if 0 <= x + 1 < self.rows and self.grid[x + 1][y] != 1:
            neighbors.append((x + 1, y))
        if 0 <= y - 1 < self.columns and self.grid[x][y - 1] != 1:
            neighbors.append((x, y - 1))
        if 0 <= y + 1 < self.columns and self.grid[x][y + 1] != 1:
            neighbors.append((x, y + 1))

        return neighbors

    def plot_grid(self, path=None):
        cmap = plt.cm.colors.ListedColormap(['black', 'red', 'white', 'yellow'])

        if path is not None:
            for position in path:
                x, y = position
                if self.grid[x][y] != 1:
                    self.grid[x][y] = 4

        plt.imshow(self.grid, cmap=cmap, interpolation='nearest')
        plt.title("City Grid with Towers and Path")
        plt.colorbar()

        if path is not None:
            for position in path:
                x, y = position
                if self.grid[x][y] != 1:
                    plt.scatter(y, x, c='blue', marker='o', s=50, edgecolors='white', linewidths=2)

        plt.show()

class CitySimulator:
    def __init__(self):
        self.city = CityGrid(10, 10, block_percentage=30, tower_cost=5, budget_limit=50)

    def simulate(self):
        self.city.optimize_tower_placement(radius=1)
        self.city.plot_grid()

        start_position = (0, 0)
        end_position = (9, 9)
        path = self.city.find_most_reliable_path(start_position, end_position)

        if path:
            self.city.plot_grid(path)
        else:
            print("Невозможно найти путь между заданными точками.")

        print(f"Потрачено: {self.city.total_spent}$, Осталось в бюджете: {self.city.budget_limit - self.city.total_spent}$")
