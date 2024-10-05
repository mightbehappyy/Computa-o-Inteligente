from random import randint
from math import floor
class particle:

    def __init__(self, dimensions: int, range: tuple):
        self.position: list[int] = []
        self.velocity: list[int] = []
        self.dimensions = dimensions
        self.range = range
        self.pbest: list [int]= []

    def generate_pos_vel(self):
        for c in range(self.dimensions):
            self.position.append(randint(self.range[0], self.range[1]))
            self.pbest = self.position
            velocity_value = randint(floor(self.range[0] * 0.3), floor(self.range[1] * 0.3))
            
            if self.position[c] + velocity_value >  self.range[1] or self.position[c] + velocity_value <  self.range[0]:
                velocity_value = -velocity_value
    
            self.velocity.append(velocity_value)

    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_pbest(self):
        return self.pbest
    
    def set_position(self, position: list):       
        for c in range(len(position)):
            if position[c] > self.range[1]:
                position[c] = self.range[1]
            elif position[c] < self.range[0]:
                position[c] = self.range[0]

        self.position = position

    def set_velocity(self, velocity):
        self.velocity = velocity
    
    def set_pbest(self, pbest):
        self.pbest = pbest