import numpy as np
import matplotlib.pyplot as plt

def part_a():
	x_m0, y_m0 = 10, 8
	x_m1, y_m1 = 6, 3

	x_t0, y_t0, d0 = 12, 4, 3.9
	x_t1, y_t1, d1 = 5, 7, 4.5

	cos = (x_t0 - x_t1) / np.sqrt((x_t0 - x_t1)**2 + (y_t0 - y_t1)**2)
	sin = (y_t1 - y_t0) / np.sqrt((x_t0 - x_t1)**2 + (y_t0 - y_t1)**2)
	rot = np.array([[cos, -sin], [sin, cos]])

	# coord transform
	x_t0_rot, _ = rot.dot(np.array([x_t0 - x_t1, y_t0 - y_t1]))

	# soln in rotated frame
	x = (x_t0_rot**2 + d1**2 - d0**2) / 2 / x_t0_rot
	y1 = np.sqrt(d1**2 - x**2)
	y2 = -y1

	# rotate back and shift origin
	rot_inv = np.array([[cos, sin], [-sin, cos]])
	x1, y1 = rot_inv.dot(np.array([x, y1])) + np.array([x_t1, y_t1])
	x2, y2 = rot_inv.dot(np.array([x, y2])) + np.array([x_t1, y_t1])

	dist_home = np.sqrt((x_m1 - x1)**2 + (y_m1 - y1)**2)
	dist_uni = np.sqrt((x_m0 - x1)**2 + (y_m0 - y1)**2)
	print(dist_uni, dist_home)

	dist_home = np.sqrt((x_m1 - x2)**2 + (y_m1 - y2)**2)
	dist_uni = np.sqrt((x_m0 - x2)**2 + (y_m0 - y2)**2)
	print(dist_uni, dist_home)

def part_b():

	x_m0, y_m0 = 10, 8
	x_m1, y_m1 = 6, 3

	x_t0, y_t0, d0, sigma0 = 12, 4, 3.9, 1
	x_t1, y_t1, d1, sigma1 = 5, 7, 4.5, np.sqrt(1.5)

	x = np.linspace(0, 15, 300)
	x, y = np.meshgrid(x, x)

	prob = 1 / 2 / np.pi / sigma0 / sigma1 * np.exp( -(np.sqrt((x - x_t0)**2 + (y - y_t0)**2) - d1)**2 / 2 / sigma1**2 
		-(np.sqrt((x - x_t1)**2 + (y - y_t1)**2) - d1)**2 / 2 / sigma1**2 )

	plt.imshow(prob, origin='lower', extent=[0, 15, 0, 15])
	plt.colorbar()
	plt.plot(x_t0, y_t0, 'o', label='Tower0')
	plt.plot(x_t1, y_t1, 'o', label='Tower1')
	plt.plot(x_m0, y_m0, 'o', label='Campus')
	plt.plot(x_m1, y_m1, 'o', label='Home')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	part_b()