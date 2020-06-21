import matplotlib.pyplot as plt
import numpy as np


def likelihood(p):
    return np.log(p / (1 - p))


def inverse_sensor_model(x, grid, z):
    l = np.zeros(grid.shape)
    res = grid[1] - grid[0]
    cell = z // res
    l[:cell] = likelihood(0.3)
    l[cell:] = likelihood(0.6)
    l[grid>(z+20)] = 0
    return l

def occupancy_map():
    res = 10
    grid = np.arange(0, 200 + res, res)
    m = 0.5 * np.ones(grid.shape)
    z = [101, 82, 91, 112, 99, 151, 96, 85, 99, 105]
    l = np.zeros(grid.shape)
    for measurement in z:
        l += inverse_sensor_model(0, grid, measurement)
    m = 1 - 1 / (1 + np.exp(l))
    plt.plot(grid, m)
    plt.show()

if __name__ == '__main__':
	occupancy_map()
