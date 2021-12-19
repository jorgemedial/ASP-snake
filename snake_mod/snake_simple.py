import pandas as pd
import numpy as np


class Snake():
    def _init_(self, x, y, lattice):
        self.x = x
        self.y = y
        self.dir_x = 1
        self.dir_y = 0
        self.lattice = lattice
        
    def move(self):
        self.x = (self.x+self.dir_x)%self.lattice.dim_x
        self.y = (self._y+self.dir_y)%self.lattice.dim_y