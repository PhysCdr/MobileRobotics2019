# Discrete Bayes filter
import numpy as np
import matplotlib.pyplot as plt

def prob(x, x_grid):

	if x == x_grid[-1]:
		p = np.array([0.75, 1])
	elif x == x_grid[0]:
		p = np.array([0.25])
	elif x == x_grid[1]:
		p = np.array([0.5, 0.25])
	else:
		p = np.array([0.25, 0.5, 0.25])
	return p

def move_forward(bel, x_grid):
	bel_new = np.zeros(x_grid.size)

	for x in x_grid:
		p = prob(x, x_grid)
		bel_new[x] = np.sum(bel[x-p.size+1:x+1] * p) # perform convolution

	return bel_new / np.sum(bel_new)

def move_back(bel, x_grid):
	bel_new = np.zeros(x_grid.size)

	for x in x_grid:
		p = prob(x, x_grid[::-1])[::-1]
		bel_new[x] = np.sum(bel[x:x+p.size] * p) # perform convolution

	return bel_new / np.sum(bel_new)

def main():
	n_grid = 20
	x_grid = np.arange(n_grid)

	# initial belief
	bel = np.zeros(n_grid)
	bel[10] = 1

	motions = [move_forward for i in range(9)]+ [move_back for i in range(3)]
	for motion in motions:
		plt.cla()
		plt.ylim([0, 1.1])
		bel = motion(bel, x_grid)
		plt.bar(x_grid, bel, label=str(np.sum(bel)))
		plt.legend()
		plt.pause(0.2)
	plt.show()

if __name__ == '__main__':
	main()