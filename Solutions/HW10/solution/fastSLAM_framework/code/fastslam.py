from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0.0
        particle['y'] = 0.0
        particle['theta'] = 0.0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0.0, 0.0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    x_y_theta = np.array(
        [[particle['x'], particle['y'], particle['theta']] for particle in particles])
    x = x_y_theta[:, 0]
    y = x_y_theta[:, 1]
    theta = x_y_theta[:, 2]
    dr1 = delta_rot1 + (noise[0] * np.abs(delta_rot1) + noise[1]
                        * np.abs(delta_trans)) * np.random.randn(len(particles))
    dr2 = delta_rot2 + (noise[0] * np.abs(delta_rot2) + noise[1]
                        * np.abs(delta_trans)) * np.random.randn(len(particles))
    dt = delta_trans + (noise[2] * np.abs(delta_trans) + noise[3] * (
        np.abs(delta_rot1) + np.abs(delta_rot2))) * np.random.randn(len(particles))
    theta += dr1
    x += dt * np.cos(theta)
    y += dt * np.sin(theta)
    theta += dr2

    for i in range(len(particles)):
        particles[i]['history'].append([particles[i]['x'], particles[i]['y']])
        particles[i]['x'] = x[i]
        particles[i]['y'] = y[i]
        particles[i]['theta'] = theta[i]


def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px)**2 + (ly - py)**2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h
    # wrt the landmark location

    H = np.zeros((2, 2))
    H[0, 0] = (lx - px) / h[0]
    H[0, 1] = (ly - py) / h[0]
    H[1, 0] = (py - ly) / (h[0]**2)
    H[1, 1] = (lx - px) / (h[0]**2)

    return h, H


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    Q_t = np.array([[1.0, 0],
                    [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                '''your code here'''
                landmark['mu'] = np.array(
                    [px + meas_range * np.cos(meas_bearing + ptheta), py + meas_range * np.sin(meas_bearing + ptheta)])
                _, H = measurement_model(particle, landmark)
                H1 = np.linalg.inv(H)
                landmark['sigma'] = H1 @ Q_t @ H1.T
                landmark['sigma'] = 0.5 * (landmark['sigma'] + landmark['sigma'].T)
                '''***        ***'''
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above.
                # calculate particle weight: particle['weight'] = ...
                '''your code here'''
                h, H = measurement_model(particle, landmark)
                Q = H @ landmark['sigma'] @ H.T + Q_t
                Q = 0.5 * (Q + Q.T)
                Q1 = np.linalg.inv(Q)
                K = landmark['sigma'] @ H.T @ Q1
                dz = (np.array([meas_range, meas_bearing]) - h)
                dz[-1] = angle_diff(meas_bearing, h[-1])
                landmark['mu'] += K @ dz
                landmark['sigma'] = (np.eye(len(landmark['sigma'])) - K @ H) @ landmark['sigma']
                landmark['sigma'] = 0.5 * (landmark['sigma'] + landmark['sigma'])
                particle['weight'] *= np.exp(-0.5 * dz.T.dot(Q1 @ dz)) / np.sqrt((2 * np.pi)**len(Q) * np.linalg.det(Q))
                '''***        ***'''
    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle
    # weights.

    new_particles = []

    '''your code here'''
    n = len(particles)
    weights = [particle['weight'] for particle in particles]
    cdf = np.cumsum(weights)
    threshold = np.random.rand() / n
    i = 0

    for j in range(n):
        ind = np.searchsorted(cdf, threshold)
        new_particles.append(copy.deepcopy(particles[ind]))
        threshold += 1 / n
    '''***        ***'''

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)

    return new_particles

def main():

    print("Reading landmark positions")
    landmarks=read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings=read_sensor_data("../data/sensor_data.dat")

    num_particles=100
    num_landmarks=len(landmarks)

    # create particle set
    particles=initialize_particles(num_particles, num_landmarks)

    # run FastSLAM
    for timestep in range(int(len(sensor_readings) / 2)):

        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        # plot filter state
        plot_state(particles, landmarks)

        # calculate new set of equally weighted particles
        particles=resample_particles(particles)

    plt.show('hold')


if __name__ == "__main__":
    main()
