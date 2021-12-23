import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import random

def direction_snake(df):
  turns = df[df['turned']==1].index
  time_between_turns = turns[1:] - turns[:-1]
  return list(time_between_turns)

class Lattice():
    def __init__(self, height, width, max_food):
        self.dim = np.array([height, width], dtype=int)
        self.max_food = max_food
        self.generate_food()

    def generate_food(self):
        self.food = np.zeros(self.dim)
        every_posible_indices = [(i, j) for i in range(self.dim[0]) for j in range(self.dim[1])]
        chosen_indices = np.random.choice(self.dim[0]*self.dim[1], size=self.max_food, replace=False)
        for chosen_index in chosen_indices:
          self.food[every_posible_indices[chosen_index]] = 1

    def show(self):
        return plt.imshow(sel.food)



class Snake():
    def __init__(self, x, y, lattice):
        self.pos = np.array([x, y], dtype=int)
        self.dir = np.array([1, 0], dtype=int)
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
        self.dir = d*self.dir@np.array([[0, 1], [-1, 0]], dtype=int)

class SnakeGame():
    def __init__(self, steps, lamda = 1/3, max_food=100, goal_food=25, height=50, turn_prob = 0.1, width=None, x=None, y=None):
        width = height if width is None else width
        x = height//5 if x is None else x
        y = (4*width)//5 if y is None else y
        
        self.max_food = max_food
        self.goal_food = goal_food
        self.lattice = Lattice(height, width, max_food)
        self.snake = Snake(x, y, self.lattice)
        self.turn_prob = turn_prob
        self.steps = steps
        self.lamda = lamda
        
        
    def play2(self):
        log = pd.DataFrame()
        i=0        
        #while self.snake.food_eaten < self.goal_food:
        while i < self.steps:
            i+=1
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

    def play(self):
        self.lattice.generate_food()
        self.snake.food_eaten = 0
        log = pd.DataFrame()
        i=0        
        while self.snake.food_eaten < self.goal_food:
        #while i < self.steps:
            T = int(np.ceil(-np.log(np.random.random_sample())/self.lamda))
            turned = 1
            spin = 1 if np.random.random_sample() < 0.5 else -1
            self.snake.change_direction(spin)
            for t in range(T+1):
              i+= 1
              there_is_food = self.snake.check_food()
              self.snake.food_eaten+= there_is_food
              self.snake.move()
              log = log.append({'pos_x': self.snake.pos[0], 'pos_y': self.snake.pos[1], 'eaten': there_is_food, 
                                'dir_x': self.snake.pos[0], 'dir_y': self.snake.dir[1], 'turned': turned, 'spin': spin}, ignore_index=True)
              turned=0
              spin=0
            
        return log


lamda_list = [3,4,5,6,7,8,9,10]

for lamda_inv in lamda_list:

    game = SnakeGame(steps=10, lamda=(1/lamda_inv), max_food=2500, goal_food=2500, turn_prob=0.127, height=50)

    cover_time = []
    time_between_turns = []

    for i in range(500):
        print(i)
        snake = game.play()
        cover_time.append(len(snake))
        time_between_turns = time_between_turns + direction_snake(snake)


    with open(f'./data/cover_time_500_{lamda_inv}.txt', 'w') as f:
        for item in cover_time:
            f.write("%s\n" % item)


    with open(f'./data/time_between_turns_500_{lamda_inv}.txt', 'w') as f:
        for item in time_between_turns:
            f.write("%s\n" % item)

# logs = [game.play() for i in range(50)]















