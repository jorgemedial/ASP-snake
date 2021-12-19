#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:52:18 2020

@author: jorge
"""

import pygame
import time
import numpy.random as rnd
import time

class Cell():
    def __init__(self, x, y, former=None):
        self.pos_x = x
        self.pos_y = y
        self.former = former

    def follow(self):
        self.pos_x = self.former.pos_x
        self.pos_y = self.former.pos_y

class Snake():
    def __init__(self, dim_x=21, dim_y = 21):
        self.head = Cell(dim_x//2, dim_y//2)
        self.body = [] #List of cells that compose the body. First cell is the last one added (Queue)
        self.dir_x = 1
        self.dir_y = 0
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.generate_mouse()
        self.t = 0.1

    def advance(self):
        for cell in self.body:
            cell.follow()
        self.head.pos_x = (self.head.pos_x+self.dir_x)%self.dim_x
        self.head.pos_y = (self.head.pos_y+self.dir_y)%self.dim_y
        self.check_mouse()
        
        

    def grow(self):
        former = self.body[0] if self.body else self.head
        cell = Cell(former.pos_x, former.pos_y, former)
        self.body.insert(0, cell)
    
    def get_positions(self):
        pos = [[cell.pos_x, cell.pos_y] for cell in self.body]
        return pos
    
    def collide(self):
        x = self.head.pos_x 
        y = self.head.pos_y
        alive =  [x,y] in self.get_positions()[1:]
        return alive 
        
    
    def generate_mouse(self):
        notAvailable = True
        while notAvailable:
            mouse_pos = rnd.randint(0, 20, 2)
            notAvailable = False
            for cell_pos in self.get_positions():
                if (mouse_pos == cell_pos).all():
                    notAvailable = True
            
        self.mouse = mouse_pos
    
    def change_direction(self, d):
        directions = {1: [-1,0], 2:[1,0], 3:[0,-1], 4: [0,1]}[d]                  
        scalar_prod = directions[0]*self.dir_x + directions[1]*self.dir_y
        if not (scalar_prod == -1):            
            self.dir_x, self.dir_y = directions
        self.advance()
    
    def check_mouse(self):
        if ([self.head.pos_x, self.head.pos_y] == self.mouse).all():
            self.generate_mouse()
            self.grow()
            self.t = self.t**1.01

class Tableau():
    def __init__(self, width, height, dim_x, dim_y):
        self.screen = pygame.display.set_mode((width, height))
        self.snake = Snake(dim_x, dim_y)
        self.block_width = width/dim_x
        self.block_height = height/dim_y
        self.bg = 25, 25, 25
    
    def get_polygon(self, coord_x, coord_y):
        polygon = [(coord_x*self.block_width,     coord_y*self.block_height),
                   ((coord_x+1)*self.block_width, coord_y*self.block_height),
                   ((coord_x+1)*self.block_width,     (coord_y+1)*self.block_height),
                   ((coord_x)*self.block_width, (coord_y+1)*self.block_height)]
        return polygon
    def draw_scene(self):
        self.screen.fill(self.bg)
        head_coord = self.get_polygon(self.snake.head.pos_x, self.snake.head.pos_y)
        pygame.draw.polygon(self.screen, (200, 200, 200), head_coord)
        for cell in self.snake.get_positions():
            coord = self.get_polygon(*cell)
            pygame.draw.polygon(self.screen, (220, 220, 200), coord)
        coord_mouse = self.get_polygon(*self.snake.mouse)
        pygame.draw.polygon(self.screen, (200, 100, 100), coord_mouse)
        pygame.display.flip()
        time.sleep(self.snake.t)
       
pygame.init()
width, height = 800, 800

list_of_evs = []
N = 20
tableau = Tableau(width, height, 20, 20)
stop = False
direction = 1

while not stop:
    ev = pygame.event.get()
    down = pygame.key.get_pressed()
    if down[pygame.K_LEFT]:
        direction = 1
    if down[pygame.K_RIGHT]:
        direction = 2
    if down[pygame.K_UP]:
        direction = 3
    if down[pygame.K_DOWN]:
        direction =  4
    if down[pygame.K_SPACE]:
        break
    
    tableau.draw_scene()
    tableau.snake.change_direction(direction)
    stop = tableau.snake.collide()
    
pygame.display.quit()
pygame.quit()
    
            
        
  
