# Satellite Program#
# ****Compute the Cartesian Location of the Vehicle****#
import numpy as np
import sys
import math


# Now convert them to decimal
def dms2rad(degrees, minutes, seconds):
    '''
    Converts degrees, minutes, seconds angles to a decimal radian angle
    :param degrees: degrees
    :type degrees number
    :param minutes: minutes
    :type minutes number
    :param seconds: seconds
    :type seconds number
    :return: number
    '''
    rad = math.pi * (
            degrees + minutes / 60 + seconds / 3600) / 180  # just convert back to decimal degrees then to radians
    return rad


def sat_pos(u, v, p, t, radius, h_s, theta_s,
            dat_pi):  # This function gives the position of the satellite provided the given parameters
    '''
    Computes geographical position of satellite
    :param u:
    :param v:
    :param p:
    :param t:
    :param radius:
    :param h_s:
    :param theta_s:
    :param dat_pi:
    :return:
    '''
    x_s = (radius + h_s) * (u * np.cos((2 * dat_pi * t) / p + theta_s) + v * np.sin(2 * dat_pi * t / p + theta_s))
    return x_s


logger = open('./satellite.log', 'w')  # start logfile to log computation

linear_sat_dat = np.zeros((24 * 9))

# height and Radius
# Get the initial Satellite data
dat_pi = np.nan
c = np.nan
R = np.nan
sid_day = np.nan

with open('./data.dat', 'r') as initial_loc:
    logger.write('\n\n--- Reading data.dat ---\n\n')
    for i, line in enumerate(initial_loc.readlines()):
        logger.flush()
        current_line = np.float64(line[1:26])
        description = line.split('/=')[1].strip()
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
            logger.write(description + ' = ' + str(current_line) + '\n')
            linear_sat_dat[i - 4] = current_line

logger.write('\n\n--- End of data.dat ---\n\n')

vehicle_input_str = sys.stdin.read().strip()
pipe_input = np.array(vehicle_input_str.split('\n'))
if pipe_input.shape[0] == 0:  # if there is no input throw an error
    raise IOError('No satellite input provided. Pipe in input from file or other executable in standard UNIX fashion')
num_steps = pipe_input.shape[0]
path_data = np.empty(shape=(num_steps, 10))

for i, line in enumerate(pipe_input):
    path_point = np.array(line.split(' '))
    if path_point.shape[0] == 10:
        path_data[i] = np.array(path_point, dtype='float')

to_be_output = []  # this will have the data from the satellites who are above the horizon appended at each step

for ns in range(num_steps):
    logger.flush()
    logger.write('\nStep: ' + str(ns) + '\n')
    prelim_path_data = path_data[ns, :]
    logger.write('Read: {:0.16} {:0.16} {:0.16} {:0.16} {:0.16} {:0.16} {:0.16} {:0.16} {:0.16} {:0.16}\n'.format(
        prelim_path_data[0], prelim_path_data[1], prelim_path_data[2], prelim_path_data[3], prelim_path_data[4],
        prelim_path_data[5], prelim_path_data[6], prelim_path_data[7], prelim_path_data[8], prelim_path_data[9]))
    # latitude angles
    deg_lat = prelim_path_data[1]
    min_lat = prelim_path_data[2]
    sec_lat = prelim_path_data[3]

    # longitude angles
    deg_lon = prelim_path_data[5]
    min_lon = prelim_path_data[6]
    sec_lon = prelim_path_data[7]

    init_sat_dat = np.reshape(linear_sat_dat, (24, 9))
    # R=6.367444500000000000*10**6;
    h_v = prelim_path_data[9]

    psi_dec = dms2rad(deg_lat, min_lat, sec_lat) * prelim_path_data[4]
    lam_dec = dms2rad(deg_lon, min_lon, sec_lon) * prelim_path_data[8]
    t_v = prelim_path_data[0]

    # Now convert these to cartesian coordinates#
    x_pos = (R + h_v) * (np.cos(psi_dec) * np.cos(lam_dec))
    y_pos = (R + h_v) * (np.cos(psi_dec) * np.sin(lam_dec))
    z_pos = (R + h_v) * np.sin(psi_dec)

    # now multiply by the rotation vector#
    rot_vec = [[np.cos(2 * np.pi * t_v / (sid_day)), -np.sin(2 * np.pi * t_v / (sid_day)), 0],
               [np.sin(2 * np.pi * t_v / (sid_day)), np.cos(2 * np.pi * t_v / (sid_day)), 0], [0, 0, 1]];
    v_pos_vec = np.matmul(rot_vec, [[x_pos], [y_pos], [z_pos]]);

    # define the sat data
    sat_step_data = np.zeros((24, 5))
    logger.write('Wrote:\n')
    for kk in range(24):
        u = init_sat_dat[kk, 0:3]
        v = init_sat_dat[kk, 3:6]
        p = init_sat_dat[kk, 6]
        h_s = init_sat_dat[kk, 7]
        theta_s = init_sat_dat[kk, 8]

        # the initial time value is just the time at the car#
        error = 1
        t_old = t_v
        while error > 3 * 10 ** (-17):
            iterator_sat_pos = sat_pos(u, v, p, t_old, R, h_s, theta_s, dat_pi)
            new_t = t_v - (np.sqrt(
                (iterator_sat_pos[0] - v_pos_vec[0]) ** 2 + (iterator_sat_pos[1] - v_pos_vec[1]) ** 2 + (
                        iterator_sat_pos[2] - v_pos_vec[2]) ** 2)) / c
            error = abs(t_old - new_t)
            t_old = new_t

        sat_signal_pos = sat_pos(u, v, p, t_old, R, h_s, theta_s, dat_pi)
        sat_step_data[kk, 0] = kk
        sat_step_data[kk, 1] = t_old
        sat_step_data[kk, 2] = sat_signal_pos[0]
        sat_step_data[kk, 3] = sat_signal_pos[1]
        sat_step_data[kk, 4] = sat_signal_pos[2]
        # now check to see if those dudes are above the horizon#
        pos_vec_dot = (
                sat_signal_pos[0] * v_pos_vec[0] + sat_signal_pos[1] * v_pos_vec[1] + sat_signal_pos[2] * v_pos_vec[2])
        v_mag_sqd = np.linalg.norm(v_pos_vec) ** 2
        # now check to see if those dudes are above the horizon#
        if pos_vec_dot > v_mag_sqd:
            logger.write('{:0.0f} {:6.16e} {:6.16e} {:6.16e} {:6.16e}\n'.format(sat_step_data[kk, 0],
                                                                                sat_step_data[kk, 1],
                                                                                sat_step_data[kk, 2],
                                                                                sat_step_data[kk, 3],
                                                                                sat_step_data[kk, 4]))
            to_be_output.append(sat_step_data[kk, :])  # if they are, append them to the array

output_array = np.squeeze(to_be_output)
for point in output_array:
    sys.stdout.write('{:0.0f} {:6.16e} {:6.16e} {:6.16e} {:6.16e}\n'.format(point[0],
                                                                            point[1],
                                                                            point[2],
                                                                            point[3],
                                                                            point[4]))

logger.close()
# End of program
