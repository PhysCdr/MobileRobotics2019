import numpy as np
import matplotlib.pyplot as plt

def pose_matrix(angle, x, y):
	return np.array([[np.cos(angle), -np.sin(angle), x],
					 [np.sin(angle), np.cos(angle), y],
					 [0, 0, 1]])

def plot_in_sensor_frame(angle, scan):
	plt.plot(angle, scan)
	plt.show()

def plot_in_global_frame(angle, distance, robo, sensor):

	# in the sensor frame
	x_l = distance * np.cos(angle)
	y_l = distance * np.sin(angle)

	robo_pose, robo_matrix = robo
	laser_pose, laser_matrix = sensor

	# in the global frame
	x_g, y_g, _ = robo_matrix.dot(laser_matrix.dot(np.array([x_l, y_l, np.ones(np.shape(y_l))])))
	theta_r, x_r, y_r = robo_pose
	theta_l, x_l, y_l = np.asarray(robo_pose) + np.asarray(laser_pose)
	delta_r = 1
	delta_l = 0.5
	plt.arrow(x_r, y_r, delta_r * np.cos(theta_r), delta_r * np.sin(theta_r), head_width = 0.1, head_length = 0.3, label='robot pose')
	plt.arrow(x_l, y_l, delta_l * np.cos(theta_l), delta_l * np.sin(theta_l), head_width = 0.1, head_length = 0.3, label='laser pose')
	plt.plot(x_g, y_g, '.', label='distance measurement')
	plt.gca().set_aspect('equal', adjustable='box')
	#plt.legend()
	plt.show()


def part_a(data):
	plot_in_sensor_frame(data[0], data[1])

def part_c(data):
	angle_robot = np.pi/4
	x_robot = 1
	y_robot = 0.5
	robo_pose = (angle_robot, x_robot, y_robot)
	robo_matrix = pose_matrix(*robo_pose)

	angle_laser = np.pi
	x_laser = 0.2
	y_laser = 0
	laser_pose = (angle_laser, x_laser, y_laser)
	laser_matrix = pose_matrix(*laser_pose)

	angle_measured = data[0]
	distance_measured = data[1]

	robo = (robo_pose, robo_matrix)
	laser = (laser_pose, laser_matrix)

	plot_in_global_frame(angle_measured, distance_measured, robo, laser)

if __name__ == '__main__':
	scan = np.loadtxt('../laserscan.dat')
	angle = np.linspace(-np.pi/2, np.pi/2, np.shape(scan)[0], endpoint=True)
	data = (angle, scan)
	#part_a(data)
	part_c(data)