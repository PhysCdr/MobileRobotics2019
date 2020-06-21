import numpy as np

def diffdrive(x, y, theta, v_l, v_r, t, l):
	v = (v_l + v_r) / 2
	w = (v_r - v_l) / l
	if np.allclose(w, 0):
		theta_n = theta
		x_n = v * np.cos(theta) * t + x
		y_n = v * np.sin(theta) * t + y
	else:
		theta_n = w * t + theta
		x_n = v / w * (np.sin(theta_n) - np.sin(theta)) + x
		y_n = v / w * (np.cos(theta) - np.cos(theta_n)) + y
	return x_n, y_n, theta_n

def part_b():
	# After reaching position x = 1.5 m, y = 2.0 m, and θ = pi/2
	# the following sequence of steering commands:
	# the robot executes
	# (a) c 1 = (v l = 0.3 m/s, v r = 0.3 m/s, t = 3s)
	# (b) c 2 = (v l = 0.1 m/s, v r = −0.1 m/s, t = 1 s)
	# (c) c 3 = (v l = 0.2 m/s, v r = 0 m/s, t = 2 s)

	x, y, theta = (1.5, 2.0, np.pi/2)
	c1 = (0.3, 0.3, 3)
	c2 = (0.1, -0.1, 1)
	c3 = (0.2, 0, 2)
	l = 0.5

	commands = [c1, c2, c3]

	for command in commands:
		x, y, theta = diffdrive(x, y, theta, *command, l)
		print(x, y, theta)

if __name__ == '__main__':
	part_b()