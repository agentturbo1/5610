#now, get all the data from the .dat file
#Satellite Program#
#****Compute the Cartesian Location of the Vehicle****#
#get the data from a particular line#
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
    rad = math.pi * (degrees + minutes / 60 + seconds / 3600) / 180 # just convert back to decimal degrees then to radians
    return(rad)


def sat_pos(u, v, p, t, Radius, h_s, theta_s, dat_pi): #T his function gives the postion of the satlelline porided the given parameters
    '''
    Computes geographical position of satelite
    :param u:
    :param v:
    :param p:
    :param t:
    :param Radius:
    :param h_s:
    :param theta_s:
    :param dat_pi:
    :return:
    '''
    x_s = (Radius + h_s) * (u * np.cos((2 * dat_pi * t) / p + theta_s) + v * np.sin(2 * dat_pi * t / p + theta_s))
    return(x_s)


vehicle_input_str = sys.stdin.read().rstrip()
pipe_input = np.array(vehicle_input_str.split('\n'))
num_steps = pipe_input.shape[0]
path_data = np.empty(shape=(num_steps, 10))

for i, line in enumerate(pipe_input):
    path_point = np.array(line.split(' '))
    if path_point.shape[0] == 10:
        path_data[i] = np.array(path_point, dtype='float')

to_be_output = [];#this will have the data from the satellites who are above the horizon appended at each step
linear_sat_dat = np.zeros((24 * 9))

# height and Radius
# Get the initial Satellite data
# initial_loc = open('./data.dat', 'r')
with open('./data.dat', 'r') as initial_loc:
    line_index = 0
    for line in initial_loc.readlines():
        current_line = np.float64(line[1:26])
        if line_index == 0:
            dat_pi = current_line
        if line_index == 1:
            c = current_line
        if line_index == 2:
            R = current_line
        if line_index == 3:
            sid_day = current_line
        # Now store all the rest of the data in a satellite array#
        if line_index > 3:
            linear_sat_dat[line_index - 4] = current_line
        line_index += 1

for ns in range(num_steps):
    prelim_path_data=path_data[ns,:];
    #latitude angles
    deg_lat=prelim_path_data[1]
    min_lat=prelim_path_data[2]
    sec_lat= prelim_path_data[3]

     #longitude angles
    deg_lon=prelim_path_data[5]
    min_lon=prelim_path_data[6]
    sec_lon= prelim_path_data[7]

    init_sat_dat = np.reshape(linear_sat_dat, (24, 9))
    # R=6.367444500000000000*10**6;
    h_v=prelim_path_data[9]

    psi_dec=dms2rad(deg_lat,min_lat,sec_lat)*prelim_path_data[4];
    lam_dec=dms2rad(deg_lon,min_lon,sec_lon)*prelim_path_data[8];
    t_v=prelim_path_data[0]

    #Now convert these to cartesian coordinates#
    x_pos=(R+h_v)*(np.cos(psi_dec)*np.cos(lam_dec));
    y_pos=(R+h_v)*(np.cos(psi_dec)*np.sin(lam_dec));
    z_pos=(R+h_v)*np.sin(psi_dec)

    #now multiply by the rotation vector#
    rot_vec=[[np.cos(2*np.pi*t_v/(sid_day)),-np.sin(2*np.pi*t_v/(sid_day)),0],[np.sin(2*np.pi*t_v/(sid_day)),np.cos(2*np.pi*t_v/(sid_day)),0],[0,0,1]];
    v_pos_vec=np.matmul(rot_vec,[[x_pos],[y_pos],[z_pos]]);

    #define the sat data
    sat_step_data=np.zeros((24,5))
    for kk in range(24):
        u=init_sat_dat[kk,0:3]
        v=init_sat_dat[kk,3:6]
        p=init_sat_dat[kk,6]
        h_s=init_sat_dat[kk,7]
        theta_s=init_sat_dat[kk,8]

        #the initial time value is just the time at the car#
        error =1;
        t_old=t_v;
        while error > 3*10**-(17):
            #new_t=t_v-(np.linalg.norm(v_pos_vec-sat_pos(u,v,p,t_old,R,h_s,theta_s,dat_pi),2))/c
            iterator_sat_pos=sat_pos(u,v,p,t_old,R,h_s,theta_s,dat_pi);
            new_t=t_v-(np.sqrt((iterator_sat_pos[0]-v_pos_vec[0])**2+(iterator_sat_pos[1]-v_pos_vec[1])**2+(iterator_sat_pos[2]-v_pos_vec[2])**2))/c
            error=abs(t_old-new_t)
            t_old=new_t;

        sat_signal_pos=sat_pos(u,v,p,t_old,R,h_s,theta_s,dat_pi)
        sat_step_data[kk,0]=kk
        sat_step_data[kk,1]=t_old
        sat_step_data[kk,2]=sat_signal_pos[0]
        sat_step_data[kk,3]=sat_signal_pos[1]
        sat_step_data[kk,4]=sat_signal_pos[2]
        #now check to see if those dudes are above the horizon#
        pos_vec_dot=(sat_signal_pos[0]*v_pos_vec[0]+sat_signal_pos[1]*v_pos_vec[1]+sat_signal_pos[2]*v_pos_vec[2])
        v_mag_sqd=np.linalg.norm(v_pos_vec)**2
        #now check to see if those dudes are above the horizon#
        if pos_vec_dot>v_mag_sqd:
            to_be_output.append(sat_step_data[kk,:])#if they are, append them to the array

output_array=np.squeeze(to_be_output)
for point in output_array:
    sys.stdout.write('{:0.0f} {:6.16e} {:6.16e} {:6.16e} {:6.16e}\n'.format(point[0],
                                                                            point[1], point[2], point[3], point[4]))
