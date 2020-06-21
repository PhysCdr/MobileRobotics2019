import numpy as np
import matplotlib.pyplot as plt
import p1

def odometry_mm(p, u, a):
	# Odometry based motion model
	# p = initial pose
	# u = odometry readings
	# a = noise params
	# returns new pose
	x, y, theta = p
	dr1_0, dr2_0, dt = u
	dr1 = dr1_0 + p1.rand_norm3(0, a[0] * np.abs(dr1_0) + a[1] * np.abs(dt))
	dr2 = dr2_0 + p1.rand_norm3(0, a[0] * np.abs(dr2_0) + a[1] * np.abs(dt))
	dt += p1.rand_norm3(0, a[2] * np.abs(dt) + a[3] * (np.abs(dr1_0) + np.abs(dr2_0)))

	theta += dr1
	x += dt * np.cos(theta)
	y += dt * np.sin(theta)
	theta += dr2
	return np.array([x, y, theta])

if __name__ == '__main__':
	p = (2.0, 4.0, 0.0)
	u = (np.pi/2, 0.0, 1.0)
	a = (0.1, 0.1, 0.01, 0.01)

	poses = np.array([odometry_mm(p, u, a) for i in range(5000)])
	x = poses[:, 0]
	y = poses[:, 1]
	plt.plot(p[0], p[1], 'o')
	plt.plot(x, y, '.')
	plt.axes().set_aspect('equal')
	plt.show()

