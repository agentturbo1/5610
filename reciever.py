import math
import random
import sys

import numpy as np


def rad2dms(rad):
    '''
    Converts decimal radians to degrees, minutes, seconds
    :param rad: radians
    :return: degrees, minutes, seconds
    :rtype: tuple
    '''
    dec_deg = rad * 180 / math.pi  # convert to a decimal form of degrees
    deg = math.floor(dec_deg)  # take the whole number part and that will be the degrees
    dec_min = (dec_deg - deg) * 60  # decimal part which we will convert to minutes and seconds
    mins = math.floor(dec_min)
    secs = (dec_min - mins) * 60
    return deg, mins, secs


logger = open('./receiver.log', 'w')

dat_pi = np.nan
c = np.nan
R = np.nan
sid_day = np.nan

with open('./data.dat', 'r') as initial_loc:
    logger.write('\n\n--- Reading data.dat ---\n\n')
    for i, line in enumerate(initial_loc.readlines()):
        current_line = np.float64(line[1:26])
        if i == 0:
            logger.writelines('pi = ' + str(current_line) + '\n')
            dat_pi = current_line
        elif i == 1:
            logger.write('c = ' + str(current_line) + '\n')
            c = current_line
        elif i == 2:
            logger.write('R = ' + str(current_line) + '\n')
            R = current_line
        elif i == 3:
            logger.write('s = ' + str(current_line) + '\n')
            sid_day = current_line
        # Now store all the rest of the data in a satellite array#
        else:  # i > 3:
            break

logger.write('\n\n--- End of data.dat ---\n\n')

# do the case where we just find four satellites, and compute the coordinates from there#
# this is the set of satellite data from one time step
# ****************#
# Call the input bm_sat_output, and then it will output write_to_output#
#
pipe_input_str = sys.stdin.read().rstrip()  # read in standard output
pipe_input = np.array(pipe_input_str.split('\n'))  # split the string by line
if pipe_input.shape[0] == 0:  # if there is no input throw an error
    raise IOError('No satellite input provided. Pipe in input from file or other executable in standard UNIX fashion')
num_signals = pipe_input.shape[0]  # number of messages received
sat_input = np.empty(shape=(num_signals, 5))  # satellite input array

for i, line in enumerate(pipe_input):  # convert all the data into float values and store in array
    sat_line = np.array(line.split(' '))
    if sat_line.shape[0] == 5:
        sat_input[i] = np.array(sat_line, dtype='float')

# Group satellite signals together for computation
all_sat_data = []
all_sat_comp = []

time = sat_input[0][1]

for hh in range(num_signals):
    current_time = sat_input[hh][1]
    if np.abs(current_time - time) > .7:
        all_sat_data.append(all_sat_comp)
        all_sat_comp = []
        time = current_time

    if hh == sat_input.shape[0] - 1:
        all_sat_data.append(all_sat_comp)

    all_sat_comp.append(sat_input[hh])

all_sat_data = np.array(all_sat_data)

# ******************#
num_time_steps = all_sat_data.shape[0]
write_to_output = []
for y in range(num_time_steps):
    logger.write('Step: ' + str(y) + '\n')
    sat_data = np.squeeze(all_sat_data[y])

    num_sats = sat_data[:, 0].shape[0]  # the number of satellites above the horizon
    # I want to take 4 of the satellites we can see from random, and use their data to compute vehicle position
    test_indices = random.sample(range(num_sats), 4)
    four_sat_data = [sat_data[test_indices[0], :], sat_data[test_indices[1], :], sat_data[test_indices[2], :],
                     sat_data[test_indices[3], :]]
    logger.write('\nSat Data:\n')
    for data in four_sat_data:
        logger.write('{:0.16} {:0.16} {:0.16} {:0.16} {:0.16}\n'.format(data[0], data[1], data[2], data[3], data[4]))

    # now we have the satellite data for that step, let's form the F matrix#
    sat1_pos = four_sat_data[0][2:5]
    sat2_pos = four_sat_data[1][2:5]
    sat3_pos = four_sat_data[2][2:5]
    sat4_pos = four_sat_data[3][2:5]
    t_sat1 = four_sat_data[0][1]
    t_sat2 = four_sat_data[1][1]
    t_sat3 = four_sat_data[2][1]
    t_sat4 = four_sat_data[3][1]

    # get the initial location of the car at time t, to start the iteration of Newton's method#
    init_b12_loc = [-1795125.28, -4477274.36, 4158393.45];  # this is the location of b12
    rot_vec = [[np.cos(2 * np.pi * t_sat1 / (86164.09)), -np.sin(2 * np.pi * t_sat1 / (86164.09)), 0],
               [np.sin(2 * np.pi * t_sat1 / (86164.09)), np.cos(2 * np.pi * t_sat1 / (86164.09)), 0], [0, 0, 1]];
    car_init_loc = np.matmul(rot_vec, init_b12_loc)

    step_num = 0
    # this is the application of Newton's method where the jacobian is formed, and then updated#
    while step_num < 100:
        Fs1 = np.linalg.norm(sat2_pos - car_init_loc) - np.linalg.norm(sat1_pos - car_init_loc) - c * (t_sat1 - t_sat2);
        Fs2 = np.linalg.norm(sat3_pos - car_init_loc) - np.linalg.norm(sat2_pos - car_init_loc) - c * (t_sat2 - t_sat3);
        Fs3 = np.linalg.norm(sat4_pos - car_init_loc) - np.linalg.norm(sat3_pos - car_init_loc) - c * (t_sat3 - t_sat4);
        F_mat = [Fs1, Fs2, Fs3]
        # Now form the Jacobian matrix#
        J11 = ((sat1_pos[0] - car_init_loc[0]) / np.linalg.norm(sat1_pos - car_init_loc)) - (
                (sat2_pos[0] - car_init_loc[0]) / np.linalg.norm(sat2_pos - car_init_loc))
        J12 = ((sat1_pos[1] - car_init_loc[1]) / np.linalg.norm(sat1_pos - car_init_loc)) - (
                (sat2_pos[1] - car_init_loc[1]) / np.linalg.norm(sat2_pos - car_init_loc))
        J13 = ((sat1_pos[2] - car_init_loc[2]) / np.linalg.norm(sat1_pos - car_init_loc)) - (
                (sat2_pos[2] - car_init_loc[2]) / np.linalg.norm(sat2_pos - car_init_loc))
        J21 = ((sat2_pos[0] - car_init_loc[0]) / np.linalg.norm(sat2_pos - car_init_loc)) - (
                (sat3_pos[0] - car_init_loc[0]) / np.linalg.norm(sat3_pos - car_init_loc))
        J22 = ((sat2_pos[1] - car_init_loc[1]) / np.linalg.norm(sat2_pos - car_init_loc)) - (
                (sat3_pos[1] - car_init_loc[1]) / np.linalg.norm(sat3_pos - car_init_loc))
        J23 = ((sat2_pos[2] - car_init_loc[2]) / np.linalg.norm(sat2_pos - car_init_loc)) - (
                (sat3_pos[2] - car_init_loc[2]) / np.linalg.norm(sat3_pos - car_init_loc))
        J31 = ((sat3_pos[0] - car_init_loc[0]) / np.linalg.norm(sat3_pos - car_init_loc)) - (
                (sat4_pos[0] - car_init_loc[0]) / np.linalg.norm(sat4_pos - car_init_loc))
        J32 = ((sat3_pos[1] - car_init_loc[1]) / np.linalg.norm(sat3_pos - car_init_loc)) - (
                (sat4_pos[1] - car_init_loc[1]) / np.linalg.norm(sat4_pos - car_init_loc))
        J33 = ((sat3_pos[2] - car_init_loc[2]) / np.linalg.norm(sat3_pos - car_init_loc)) - (
                (sat4_pos[2] - car_init_loc[2]) / np.linalg.norm(sat4_pos - car_init_loc))
        J_Mat = [[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]]
        Jinv = np.linalg.inv(J_Mat);
        diff_term = np.matmul(Jinv, F_mat)
        new_k = car_init_loc - diff_term
        step_num = step_num + 1;
        car_init_loc = new_k

        # now, we want to convert the location to degree minute and second
        # R = 6.367444500000000000 * 10 ** 6;

    h = np.sqrt(car_init_loc[0] ** 2 + car_init_loc[1] ** 2 + car_init_loc[2] ** 2) - R
    x = car_init_loc[0]
    y = car_init_loc[1]
    z = car_init_loc[2]

    # HERE IS WHERE WE CONVERT THE CARTESIAN COORDINATES TO OUTPUT FORM#
    # now we get psi#
    if x ** 2 + y ** 2 != 0:
        psi = np.arctan(z / np.sqrt(x ** 2 + y ** 2))
    if x == 0 and y == 0 and z > 0:
        psi = np.pi / 2
    if x == 0 and y == 0 and z < 0:
        psi = -1 * np.pi / 2
    NS = 1;
    if psi < 0:
        NS = -1
        psi = np.abs(psi)

    # now we get lambda#
    if x > 0 and y > 0:
        lam = np.arctan(y / x)
    if x < 0 < y:
        lam = np.pi + np.arctan(y / x)
    if x > 0 > y:
        lam = 2 * np.pi + np.arctan(y / x)

    # lam_dms=dec2dms(lam)
    psi_dms = rad2dms(psi)

    # Now we have lambda at t=0, but... we're not at t=0, so we need to get that#

    # what is the time at the car?#

    car_time = ((np.linalg.norm(sat2_pos - car_init_loc)) / c) + t_sat2;
    extra_rot = (car_time / sid_day) * np.pi * 2
    adj_lam = lam - extra_rot
    # now we have lam make sure it is between 0 and pi#
    if lam > 0 and lam < np.pi:
        EW = 1
    if lam < 0 and np.abs(lam) < np.pi:
        EW = -1
        lam = np.abs(lam)
    if lam < 0 and np.abs(lam) > np.pi:
        EW = 1
        lam = 2 * np.pi - np.abs(lam)
    if lam > np.pi:
        EW = -1;
        lam = 2 * np.pi - lam
    lam_dms = rad2dms(np.abs(adj_lam))

    to_be_output = [car_time, psi_dms[0], psi_dms[1], psi_dms[2], NS, lam_dms[0], lam_dms[1], lam_dms[2], EW, h]
    write_to_output.append(to_be_output)

logger.close()
# End of program


