import numpy as np
#import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

#plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(np.abs(chisquare_scale*max_eigval))
    height = 2 * np.sqrt(np.abs(chisquare_scale*min_eigval))
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    mu[0] = x + delta_trans * np.cos(theta + delta_rot1)
    mu[1] = y + delta_trans * np.sin(theta + delta_rot1)
    mu[2] = theta + delta_rot1 + delta_rot2

    Qt = np.eye(3) * 0.2
    Qt[-1,-1] = 0.02
    Gt = np.eye(3)
    Gt[0,-1] = -delta_trans * np.sin(theta + delta_rot2)
    Gt[1,-1] = delta_trans * np.cos(theta + delta_rot2)
    sigma = Gt @ sigma @ Gt.T + Qt
    return mu, sigma

def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    landmark_x = np.array([landmarks[landmark_id][0] for landmark_id in ids])
    landmark_y = np.array([landmarks[landmark_id][1] for landmark_id in ids])
    Ht_T = np.array([x - landmark_x, y - landmark_y, np.zeros(landmark_y.shape)])
    distances = np.sqrt((x - landmark_x)**2 + (y - landmark_y)**2)
    Ht_T /= distances
    Rt = np.eye(len(ranges)) / 2
    Kt = sigma @ Ht_T @ np.linalg.inv(Ht_T.T @ sigma @ Ht_T + Rt)

    mu += Kt @ (ranges - distances)
    sigma = (np.eye(sigma.shape[0]) - Kt @ Ht_T.T) @ sigma
    
    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    #run kalman filter
    for timestep in range(int(len(sensor_readings)/2)):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show('hold')

if __name__ == "__main__":
    main()