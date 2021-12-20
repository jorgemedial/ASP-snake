import pandas as pd
import numpy as np

class Lattice():
    def __init__(self, height, width, max_food):
        self.dim = np.array([height, width])
        self.food = np.random.randint(2, size=[height, width])


class Snake():
    def __init__(self, x, y, lattice):
        self.pos = np.array([x, y])
        self.dir = np.array([1, 0])
        self.lattice = lattice
        self.food_eaten = 0
        
    def move(self):
        self.pos = (self.pos + self.dir)%self.lattice.dim

    def check_food(self):
        if (self.lattice.food[tuple(self.pos)] == 1):
            self.lattice.food[tuple(self.pos)] = 0
            there_is_food = 1
        else:
            there_is_food = 0
        return there_is_food

    def change_direction(self, d):
        self.dir = d*self.dir@np.array([[0, 1], [-1, 0]])

class SnakeGame():
    def __init__(self, max_food=100, goal_food=25, height=50, turn_prob = 0.1, width=None, x=None, y=None):
        width = height if width is None else width
        x = height//5 if x is None else x
        y = (4*width)//5 if y is None else y
        
        self.max_food = max_food
        self.goal_food = goal_food
        self.lattice = Lattice(height, width, max_food)
        self.snake = Snake(x, y, self.lattice)
        self.turn_prob = turn_prob
        self.log = self.play()
        
        
    def play(self):
        log = pd.DataFrame()        
        while self.snake.food_eaten < self.goal_food:
            there_is_food = self.snake.check_food()
            self.snake.food_eaten+= there_is_food
            self.snake.move()
            turned = 0
            spin = 0
            if np.random.random_sample() < self.turn_prob:
                turned = 1
                spin = 1 if np.random.random_sample() < 0.5 else -1
                self.snake.change_direction(spin)
            
            log = log.append({'pos_x': self.snake.pos[0], 'pos_y': self.snake.pos[1], 'eaten': there_is_food, 
                        'dir_x': self.snake.pos[0], 'dir_y': self.snake.dir[1], 'turned': turned, 'spin': spin}, ignore_index=True)
        return log