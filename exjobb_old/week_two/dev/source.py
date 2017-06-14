import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class Room:

    def __init__(self, north_wall, west_wall, south_wall, east_wall):
        self.north_wall = north_wall
        self.west_wall = west_wall
        self.south_wall = south_wall
        self.east_wall = east_wall

    def get_walls(self):
        return (self.north_wall, self.west_wall, self.south_wall, self.east_wall)

    def __str__(self):
        return str("Room with four walls")


class Wall:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def set_value(self,value):
        self.value = value

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name

    def __str__(self):
        return "name: %s, value: %d" % (self.name, self.value)


class Boundary:

    def __init__(self,boundary_type):
        self.boundary_type = boundary_type
        self.value = []

    def _set_value(self,value):
        self.value = value

    def get_value(self):
        if self.boundary_type == "Neumann":
            return 0
        if self.boundary_type == "Dirichlet":
            return 0


class Solver:

    def __init__(self,problem,iterations=40,relaxation=0.8,Ni=40,Nj=79):
        self.problem = problem
        self.iterations = iterations
        self.relaxation = relaxation
        self.Ni = Ni
        self.Nj = Nj
        self.ni = self.Ni-2
        self.nj = self.Nj-2
        self.dx = 1 / (min(self.Ni,self.Nj)+1)
        self.solution = []

    def get_a(self):
        Ni = self.ni
        Nj = self.nj
        sup = np.ones(Ni * Nj - 1)
        for i in range(1, len(sup) + 1):
            if i % Ni == 0: sup[i - 1] = 0
        A = np.diag(np.ones(Ni * Nj - Ni), -Ni) \
            + np.diag(sup, -1) \
            + np.diag(-4. * np.ones(Ni * Nj), 0) \
            + np.diag(sup, 1) \
            + np.diag(np.ones(Ni * Nj - Ni), Ni)
        return sparse.csr_matrix(1 / (self.dx ** 2) * A)

    def get_boundary(self,walls):
        b = np.zeros((self.ni, self.nj))
        for wall in walls:
            if wall.get_name() == "North":
                b[-1, :] += wall.get_value()
            if wall.get_name() == "West":
                b[:, 0] += wall.get_value()
            if wall.get_name() == "South":
                b[0, :] += wall.get_value()
            if wall.get_name() == "East":
                 b[:, -1] += wall.get_value()

        return np.asarray(b).reshape(-1)/(self.dx ** 2)

    def visualize(self):
        plt.pcolor(self.solution)
        plt.colorbar()
        plt.show()

    def run(self):
        walls = self.problem.get_walls()
        A = self.get_a()
        b = self.get_boundary(walls)

        solution = np.reshape(scipy.sparse.linalg.spsolve(A, -1*b), (self.ni, self.nj))

        # reshape for plot
        u = np.zeros((self.Ni,self.Nj))
        u[1:-1, 1:-1] = solution
        u[0, :] = walls[2].get_value()
        u[:, 0] = walls[1].get_value()
        u[:, -1] = walls[3].get_value()
        u[-1, :] = walls[0].get_value()
        self.solution = u



