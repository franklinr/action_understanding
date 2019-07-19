###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 140717/170717/200717/120817/1407817/180917/050918
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%--------------------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
# Some header and necessary function import
import numpy as np
import time
import matplotlib as plt
from IPython.display import clear_output
from sklearn.preprocessing import normalize
##############################################################################################

import os
import sys

ss_lib_path = 'SS_PYLIBS/'
sys.path.insert(0, ss_lib_path)
#---------------------------------------------------------------------------------------------

from ss_computation import *
from ss_image_video import *
from ss_input_output import *
from ss_drawing import *

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_take_velocity_direction(int_velo, theta_thresh, velo_theta = [], num_theta_div = 10.0):

    if(not(np.isscalar(velo_theta))):
        velo_theta = np.mod(np.arctan2(int_velo[1, 0], int_velo[0, 0])*(180/np.pi), 360.0)

    int_theta = velo_theta - theta_thresh/2.0
    end_theta = velo_theta + theta_thresh/2.0
    if(not(np.isscalar(num_theta_div))):
        theta_div = 0.0
    else:
        theta_div = theta_thresh/num_theta_div

    if(theta_div == 0.0):
        velo_theta = np.array([int_theta, end_theta])
    else:
        velo_theta = np.arange(int_theta, end_theta, theta_div)
    if velo_theta.size == 0:
        velo_theta = np.array([int_theta, end_theta])
    velo_theta = np.mod(velo_theta, 360.0)
    
    velo_theta, idt = ss_take_sample(velo_theta, 'random', 1, 1)
    Temp_pos_theta = velo_theta*(np.pi/180.0)
    
    new_velo_dirc = np.zeros((2, 1))
    new_velo_dirc[0,:] = np.cos(Temp_pos_theta)
    new_velo_dirc[1,:] = np.sin(Temp_pos_theta)

    return(new_velo_dirc)


def ss_take_wolf_velocity_direction(int_velo, theta_thresh, velo_theta = [], sample_flag='uniform'):

    if(not(np.isscalar(velo_theta))):
        velo_theta = np.mod(np.arctan2(int_velo[1, 0], int_velo[0, 0])*(180/np.pi), 360.0)

    int_theta = velo_theta - theta_thresh/2.0
    end_theta = velo_theta + theta_thresh/2.0
    
    velo_theta = np.array([int_theta, end_theta])
    velo_theta = np.mod(velo_theta, 360.0)
#     print(velo_theta)
    if sample_flag == 'random':
        velo_theta, idt = ss_take_sample(velo_theta, 'random', 1, 1)
    else:
        velo_theta = velo_theta[0]
#     print(velo_theta)
    Temp_pos_theta = velo_theta*(np.pi/180.0)
    new_velo_dirc = np.zeros((2, 1))
    new_velo_dirc[0,:] = np.cos(Temp_pos_theta)
    new_velo_dirc[1,:] = np.sin(Temp_pos_theta)

    return(new_velo_dirc)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_init_velocity_direction(num_points, int_theta = 0.0, end_theta = 360.0, theta_div = 1.0):

    theta = np.arange(int_theta, end_theta, theta_div) # velocity direction range
    velo_theta, idt = ss_take_sample(theta, 'random', num_points, 1) # random velocity direction
    int_pos_velo = np.zeros((2,num_points))
    Temp_pos_theta = velo_theta*(np.pi/180.0)
    int_pos_velo[0,:] = np.cos(Temp_pos_theta)
    int_pos_velo[1,:] = np.sin(Temp_pos_theta)

    return(int_pos_velo)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_boundary_constraints(pos, pos_velo, row, col, dx_row, dy_col, max_speed, max_steer_force):

    steer = np.zeros((pos.shape))
    des_dirc = np.zeros((pos.shape))
    Temp_des_dirc = np.zeros((pos.shape[0], 4))
    Temp_des_dirc1 = Temp_des_dirc.copy()

    x = pos[0,0].copy()
    y = pos[1,0].copy()
    vx = pos_velo[0,0].copy()
    vy = pos_velo[1,0].copy()
    des_dirc_count = 0
    if(x < dx_row):
        if(vy == 0):
            Temp_dirc = np.array([[max_speed], [max_speed]]).copy()
        elif(abs(vy) > 1.0):
            Temp_dirc = np.array([[max_speed], [vy]]).copy()
        else:
            Temp_dirc = np.array([[max_speed], [(max_speed)*np.sign(vy)]]).copy()
        Temp_des_dirc[:,0] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1
    if(x > row-dx_row-1):
        if(vy == 0):
            Temp_dirc = np.array([[-max_speed], [max_speed]]).copy()
        elif(abs(vy) > 1.0):
            Temp_dirc = np.array([[-max_speed], [vy]]).copy()
        else:
            Temp_dirc = np.array([[-max_speed], [(max_speed)*np.sign(vy)]]).copy()
        Temp_des_dirc[:,1] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1

    if(y < dy_col):
        if(vx == 0):
            Temp_dirc = np.array([[max_speed], [max_speed]]).copy()
        elif(abs(vx) > 1.0):
            Temp_dirc = np.array([[vx], [max_speed]]).copy()
        else:
            Temp_dirc = np.array([[(max_speed)*np.sign(vx)], [max_speed]]).copy()
        Temp_des_dirc[:,2] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1
    if(y > col-dy_col-1):
        if(vx == 0):
            Temp_dirc = np.array([[max_speed], [-max_speed]]).copy()
        elif(abs(vx) > 1.0):
            Temp_dirc = np.array([[vx], [-max_speed]]).copy()
        else:
            Temp_dirc = np.array([[(max_speed)*np.sign(vx)], [-max_speed]]).copy()
        Temp_des_dirc[:,3] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1

    if(des_dirc_count):
        Temp_des_dirc1 = Temp_des_dirc.copy()
        Temp_des_dirc = normalize(Temp_des_dirc, axis=0)
        des_dirc = np.reshape(np.sum(Temp_des_dirc.copy(), axis=1), pos.shape)
        des_dirc = normalize(des_dirc, axis=0)
        des_dirc = max_speed*des_dirc

        steer = des_dirc - pos_velo
        steer = normalize(steer, axis=0)
        steer = max_steer_force*steer

    return(steer, des_dirc, Temp_des_dirc, Temp_des_dirc1)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_neighbour_constraints(pos, pos_velo, neighbour_pos, neighbour_pos_velo, dist_thresh, max_speed, max_steer_force):

    steer = np.zeros((pos.shape))
    des_dirc = np.zeros(pos.shape)
    Temp_des_dirc = np.zeros(neighbour_pos.shape)
    Temp_des_dirc1 = Temp_des_dirc.copy()
    num_neighbour_objects = neighbour_pos.shape[1]

    dist_pos_comp_pos = ss_euclidean_dist(pos, neighbour_pos, 1)
    des_dirc_count = 0
    for snno in range(num_neighbour_objects):
        if((dist_pos_comp_pos[0,snno] > 0.0) & (dist_pos_comp_pos[0,snno] < dist_thresh)):
            Temp_diff = pos - np.reshape(neighbour_pos[:,snno], pos.shape)
            Temp_des_dirc1[:,snno] = np.reshape(Temp_diff.copy(), (2,))
            Temp_diff = normalize(Temp_diff, axis=0)#/dist_pos_comp_pos[0,snno]
            Temp_des_dirc[:,snno] = np.reshape(Temp_diff.copy(), (2,))
            des_dirc_count += 1

    if(des_dirc_count):
        des_dirc = np.reshape(np.sum(Temp_des_dirc.copy(), axis=1), pos.shape)
        des_dirc = normalize(des_dirc, axis=0)
        des_dirc = max_speed*des_dirc

        steer = des_dirc - pos_velo
        steer = normalize(steer, axis=0)
        steer = max_steer_force*steer

    return(steer, des_dirc, Temp_des_dirc, Temp_des_dirc1)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_border_neighbour_constraints(pos, pos_velo, row, col, dx_row, dy_col, neighbour_pos, neighbour_pos_velo, neighbour_disjoint_threshold, max_speed, combined_max_steer_force):

    steer = np.zeros((pos.shape))
    des_dirc = np.zeros((pos.shape))
    num_neighbour_objects = neighbour_pos.shape[1]
    Temp_des_dirc = np.zeros((pos.shape[0], 4+num_neighbour_objects))
    Temp_des_dirc1 = Temp_des_dirc.copy()

    x = pos[0,0].copy()
    y = pos[1,0].copy()
    vx = pos_velo[0,0].copy()
    vy = pos_velo[1,0].copy()
    des_dirc_count = 0
    if(x < dx_row):
        if(vy == 0):
            Temp_dirc = np.array([[max_speed], [max_speed]]).copy()
        elif(abs(vy) > 1.0):
            Temp_dirc = np.array([[max_speed], [vy]]).copy()
        else:
            Temp_dirc = np.array([[max_speed], [(max_speed)*np.sign(vy)]]).copy()

        Temp_des_dirc[:,0] = np.reshape(Temp_dirc.copy(), (2,))
        Temp_des_dirc1[:,0] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1
    if(x > row-dx_row-1):
        if(vy == 0):
            Temp_dirc = np.array([[-max_speed], [max_speed]]).copy()
        elif(abs(vy) > 1.0):
            Temp_dirc = np.array([[-max_speed], [vy]]).copy()
        else:
            Temp_dirc = np.array([[-max_speed], [(max_speed)*np.sign(vy)]]).copy()
        Temp_des_dirc[:,1] = np.reshape(Temp_dirc.copy(), (2,))
        Temp_des_dirc1[:,1] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1

    if(y < dy_col):
        if(vx == 0):
            Temp_dirc = np.array([[max_speed], [max_speed]]).copy()
        elif(abs(vx) > 1.0):
            Temp_dirc = np.array([[vx], [max_speed]]).copy()
        else:
            Temp_dirc = np.array([[(max_speed)*np.sign(vx)], [max_speed]]).copy()
        Temp_des_dirc[:,2] = np.reshape(Temp_dirc.copy(), (2,))
        Temp_des_dirc1[:,2] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1
    if(y > col-dy_col-1):
        if(vx == 0):
            Temp_dirc = np.array([[max_speed], [-max_speed]]).copy()
        elif(abs(vx) > 1.0):
            Temp_dirc = np.array([[vx], [-max_speed]]).copy()
        else:
            Temp_dirc = np.array([[(max_speed)*np.sign(vx)], [-max_speed]]).copy()
        Temp_des_dirc[:,3] = np.reshape(Temp_dirc.copy(), (2,))
        Temp_des_dirc1[:,3] = np.reshape(Temp_dirc.copy(), (2,))
        des_dirc_count += 1

    dist_pos_comp_pos = ss_euclidean_dist(pos, neighbour_pos, 1)
    for snno in range(num_neighbour_objects):
        if((dist_pos_comp_pos[0,snno] > 0.0) & (dist_pos_comp_pos[0,snno] < neighbour_disjoint_threshold)):
            Temp_diff = pos - np.reshape(neighbour_pos[:,snno], pos.shape)
            Temp_des_dirc1[:,4+snno] = np.reshape(Temp_diff.copy(), (2,))
            Temp_diff = normalize(Temp_diff, axis=0)#/dist_pos_comp_pos[0,snno]
            Temp_des_dirc[:,4+snno] = np.reshape(Temp_diff.copy(), (2,))
            des_dirc_count += 1
    if(des_dirc_count):
        Temp_des_dirc = normalize(Temp_des_dirc, axis=0)
        des_dirc = np.reshape(np.sum(Temp_des_dirc.copy(), axis=1), pos.shape)
        des_dirc = normalize(des_dirc, axis=0)
        des_dirc = max_speed*des_dirc

        steer = des_dirc - pos_velo
        steer = normalize(steer, axis=0)
        steer = combined_max_steer_force*steer

    return(steer, des_dirc, Temp_des_dirc, Temp_des_dirc1)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_check_direction(pos, row, col, dx_row, dy_col, neighbour_pos, neighbour_disjoint_threshold):

    num_neighbour_objects = neighbour_pos.shape[1]
    x = pos[0,0].copy()
    y = pos[1,0].copy()

    dirc_change_flag = 0
    if(x < dx_row):
        dirc_change_flag += 1
    if(x > row-dx_row-1):
        dirc_change_flag += 1

    if(y < dy_col):
        dirc_change_flag += 1
    if(y > col-dy_col-1):
        dirc_change_flag += 1

    dist_pos_comp_pos = ss_euclidean_dist(pos, neighbour_pos, 1)
    for snno in range(num_neighbour_objects):
        if((dist_pos_comp_pos[0,snno] > 0.0) & (dist_pos_comp_pos[0,snno] < neighbour_disjoint_threshold)):
            dirc_change_flag += 1

    return(dirc_change_flag)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_check_bounday(pos, row, col, dx_row, dy_col):

    new_pos = pos.copy()
    
    if(pos[0,0] < dx_row):
        new_pos[0,0] = dx_row
    if(pos[0,0] > row-dx_row-1):
        new_pos[0,0] = row-dx_row-1

    if(pos[1,0] < dy_col):
        new_pos[1,0] = dy_col
    if(pos[1,0] > col-dy_col-1):
        new_pos[1,0] = col-dy_col-1
        
    return new_pos
     
def ss_chasing_random_trajectory(int_pos, int_pos_velo, max_speed, max_steer_force, row, column, dx_row, dx_column, tajectory_length, velo_theta_thresh, disjoint_threshold):

    #how frequent object motion direction (AOM) will be changed. It depends on the chasing subtlety (CS) 
    # If CS is low then AOM will change less frequent and vise versa
    dirc_change_flag = np.round(np.sqrt(2.2*np.sqrt(360.0)) - np.sqrt(np.sqrt(velo_theta_thresh[0, 1])))#6#np.round(1.3*np.sqrt(360.0) - np.sqrt(velo_theta_thresh[0, 1]))

    num_points = int_pos.shape[1]
    pos = np.zeros((2, num_points, tajectory_length)) #variable for object positions
    pos_velo = np.zeros((2, num_points, tajectory_length)) #variable for object velocity direction
    neighbour_steer_dirc = np.zeros((2, num_points, tajectory_length)) #variable for steering direction (based on neighbour objects)
    boundary_steer_dirc = np.zeros((2, num_points, tajectory_length))#variable for steering direction (based on boundary condition)

    pos[:,:,0] = int_pos
    pos_velo[:,:,0] = int_pos_velo
    
    avegrate_max_speed = np.mean(max_speed)
    avegrate_min_speed = np.mean(max_speed)/2

    #################################

    for snt in range(1, tajectory_length):
        # print('F#-',snt)

        sheep_pos = np.reshape(pos[:,0,snt-1], (pos.shape[0], 1)).copy()
        wolf_pos = np.reshape(pos[:,1,snt-1], (pos.shape[0], 1)).copy()
        wolf_sheep_angle = np.mod(ss_anglefrom2points(wolf_pos, sheep_pos)*(180/np.pi), 360)

        for snp in range(num_points):
            prev_pos_all = np.reshape(pos[:,:,snt-1], (pos.shape[0], pos.shape[1])).copy()
            prev_pos = np.reshape(pos[:,snp,snt-1], (pos.shape[0], 1)).copy()
            prev_pos_velo_all = np.reshape(pos_velo[:,:,snt-1], (pos_velo.shape[0], pos_velo.shape[1])).copy()
            prev_pos_velo = np.reshape(pos_velo[:,snp,snt-1], (pos_velo.shape[0], 1)).copy()
            boundary_max_steer_force = max_steer_force[0, snp].copy()
            neighbour_max_steer_force = max_steer_force[0, snp].copy()
            combined_max_steer_force = 1.2*max_steer_force[0, snp].copy()

            neighbour_disjoint_threshold = disjoint_threshold[0, snp].copy()
            boundary_dx_row = disjoint_threshold[0, snp].copy()#dx_row
            boundary_dx_column = disjoint_threshold[0, snp].copy()#dx_column
            
            if(snp == 0):
                neighbour_disjoint_threshold = 1.*disjoint_threshold[0, snp].copy()
                boundary_dx_row = 1.2*boundary_dx_row
                boundary_dx_column = 1.2*boundary_dx_column

            dirc_change_flag1 = ss_check_direction(prev_pos, row, column, boundary_dx_row, boundary_dx_column, prev_pos_all, neighbour_disjoint_threshold)
            if(dirc_change_flag1 == 0.0):
                if(np.mod(snt, dirc_change_flag+snp) == 0):#change the AOM dependns on the object number as well
                    if((snp == 1)):
                        prev_pos_velo = prev_pos_velo#max_speed[0, snp]*ss_take_velocity_direction(prev_pos_velo.copy(), velo_theta_thresh[0, snp-1].copy()) #take random velocity direction
                    else:
                        prev_pos_velo = max_speed[0, snp]*ss_take_velocity_direction(prev_pos_velo.copy(), velo_theta_thresh[0, snp].copy(), [], velo_theta_thresh[0, snp].copy()) #take random velocity direction
           
            if(snp == 1):
                prev_pos_all[:,0] = prev_pos_all[:,1].copy()# To avoid sheep destruction

                if(velo_theta_thresh[0, snp] <= 180.0):#initially 180
                    wolf_dirc_change_flag = 1
                else:
                    wolf_dirc_change_flag = np.round((np.sqrt(1.5*velo_theta_thresh[0, snp])))#initially 1.5
                if(np.mod(snt, wolf_dirc_change_flag) == 0):
                    prev_pos_velo = ss_take_wolf_velocity_direction([], velo_theta_thresh[0, snp].copy(), wolf_sheep_angle)
                    prev_pos_velo = normalize(prev_pos_velo, axis=0)
                    prev_pos_velo = max_speed[0, snp]*prev_pos_velo
                    
            Temp_boundary_steer_dirc, Temp_boundary_dirc, Temp_boundary_des_dirc, Temp_boundary_des_dirc1 = ss_border_neighbour_constraints(prev_pos, prev_pos_velo, row, column, boundary_dx_row, boundary_dx_column, prev_pos_all, prev_pos_velo_all, neighbour_disjoint_threshold, max_speed[0, snp], combined_max_steer_force)
            Temp_neighbour_steer_dirc = np.zeros((2,1))

            new_pos_accl = Temp_neighbour_steer_dirc + Temp_boundary_steer_dirc
            new_pos_velo = prev_pos_velo + new_pos_accl
            new_pos_velo = normalize(new_pos_velo, axis=0)
            new_max_speed = max_speed[0, snp].copy()
            wolf_sheep_distance = ss_euclidean_dist(pos[:,0,snt-1], pos[:,1,snt-1], 1)
        
            if(snp == 1):
                if(wolf_sheep_distance < neighbour_disjoint_threshold):
                    new_max_speed -= np.exp(-np.sqrt(np.sqrt(velo_theta_thresh[0, snp]/20)))#2
                    new_max_speed = max(avegrate_min_speed, new_max_speed)#max(3, new_max_speed-1) #270717
                    max_speed[0, snp] = new_max_speed
                else:
                    new_max_speed = min(avegrate_max_speed, new_max_speed+1)#min(5, new_max_speed+1)#270717
                    max_speed[0, snp] = new_max_speed

            new_pos_velo = new_max_speed*new_pos_velo
            new_pos = np.round(prev_pos + new_pos_velo)
            new_pos = ss_check_bounday(new_pos, row, column, dx_row/2., dx_column/2.)
            
            pos[:,snp,snt] = np.reshape(new_pos, (new_pos.shape[0],)).copy()
            pos_velo[:,snp,snt] = np.reshape(new_pos_velo, (new_pos_velo.shape[0],)).copy()
            


    return pos, pos_velo


def ss_pushing_id(neighbour_pos, row, col, dx_row, dy_col):

    Temp_dist = ss_euclidean_dist(neighbour_pos, neighbour_pos, 1)
    np.fill_diagonal(Temp_dist, np.max(Temp_dist)+2)
    #Temp_dist_sort = np.sort(Temp_dist, axis = None)
    #min_idx = np.where(Temp_dist == Temp_dist_sort[np.int(np.prod(Temp_dist.shape)/4)])
    min_idx = np.where(Temp_dist == Temp_dist.min())
    pushing_id = np.array([min_idx[0][0], min_idx[1][0]])

    return(pushing_id)


def ss_pushing_random_trajectory(int_pos, int_pos_velo, pushing_subtlety, 
                                 num_pushing, pushing_contact_distance, pushing_delay, 
                                 push_interval, max_speed, max_steer_force, 
                                 row, column, dx_row, dx_column, object_radius, tajectory_length, 
                                 velo_theta_thresh, disjoint_threshold):

    pusher_halt = 10
    dirc_change_flag = 5
    num_points = int_pos.shape[1]
    pos = np.zeros((2, num_points, tajectory_length)) #variable for object positions
    pos_velo = np.zeros((2, num_points, tajectory_length)) #variable for object velocity direction
    neighbour_steer_dirc = np.zeros((2, num_points, tajectory_length)) #variable for steering direction (based on neighbour objects)
    boundary_steer_dirc = np.zeros((2, num_points, tajectory_length))#variable for steering direction (based on boundary condition)

    pos[:,:,0] = int_pos
    pos_velo[:,:,0] = int_pos_velo
    
    avegrate_max_speed = np.mean(max_speed)
    avegrate_min_speed = np.mean(max_speed)/2
    
    step_size = (tajectory_length - 2*push_interval - (push_interval+pushing_delay+pusher_halt)*num_pushing)/(num_pushing)
    T = np.array(np.arange(push_interval, tajectory_length - (push_interval+pushing_delay+pusher_halt)-step_size, step_size+push_interval))
    push_frame_info = np.zeros((3, num_pushing))
    push_frame_info[0,:] = T.copy()
    push_frame_info[1,:] = push_frame_info[0,:] + step_size + push_interval - 1
    push_count = 0
    pushing_start_flag = 1
    pushing_halt_flag = 0
    pushing_delay_count = 0
    pushing_end_flag = 0
    
    pusher_halt_count = 0
    push_id = np.zeros((2, num_pushing))# np.array([[0, 1, 2], [1, 2, 3]])
    push_id = push_id.astype('int')

    #################################

    for snt in range(1, tajectory_length):
        # print('F#-',snt)

        for snp in range(num_points):
            
            prev_pos_all = np.reshape(pos[:,:,snt-1], (pos.shape[0], pos.shape[1])).copy()
            prev_pos = np.reshape(pos[:,snp,snt-1], (pos.shape[0], 1)).copy()
            prev_pos_velo_all = np.reshape(pos_velo[:,:,snt-1], (pos_velo.shape[0], pos_velo.shape[1])).copy()
            prev_pos_velo = np.reshape(pos_velo[:,snp,snt-1], (pos_velo.shape[0], 1)).copy()
            boundary_max_steer_force = max_steer_force[0, snp].copy()
            neighbour_max_steer_force = max_steer_force[0, snp].copy()
            combined_max_steer_force = 1.2*max_steer_force[0, snp].copy()
            neighbour_disjoint_threshold = disjoint_threshold[0, snp].copy()
            boundary_dx_row = disjoint_threshold[0, snp].copy()#dx_row
            boundary_dx_column = disjoint_threshold[0, snp].copy()#dx_column
            new_max_speed = max_speed[0, snp].copy()
##############PUISHING LOGIC####################
            if(push_count < num_pushing):
                if((snt >= push_frame_info[0, push_count]) and (snt < push_frame_info[1, push_count])):
                    
                    if(snt == push_frame_info[0, push_count]):
                        Temp_push_id = ss_pushing_id(prev_pos_all.copy(), row, column, boundary_dx_row, boundary_dx_column)
                        push_id[:, push_count] = Temp_push_id.copy()
                        
                        pusher_pos = np.reshape(prev_pos_all[:,push_id[0, push_count]], (prev_pos_all.shape[0], 1)).copy()
                        pushee_pos = np.reshape(prev_pos_all[:,push_id[1, push_count]], (prev_pos_all.shape[0], 1)).copy()
                        Temp_dist = np.linalg.norm(pusher_pos - pushee_pos)
                        pusher_pushee_dist_count = Temp_dist//ss_denominator_check(max_speed[0, push_id[0, push_count]]) + 1
                        
                        pusher_speed_during_pushing = Temp_dist/pusher_pushee_dist_count     
                        
                    #FOR PUSHER MOTION
                    if(snp == push_id[0, push_count]): 
                        if(pushing_start_flag):#start pushing
                            pusher_pos = np.reshape(pos[:,push_id[0, push_count],snt-1], (pos.shape[0], 1)).copy()
                            pushee_pos = np.reshape(pos[:,push_id[1, push_count],snt-1], (pos.shape[0], 1)).copy()
                            Temp_dist = np.linalg.norm(pusher_pos - pushee_pos)
                            #if(Temp_dist >= 2*object_radius + 2*max_speed[0, push_id[0, push_count]] + pushing_contact_distance):
                            if(Temp_dist > 2*object_radius+2 + pushing_contact_distance):
                                pusher_pushee_angle = np.mod(ss_anglefrom2points(pusher_pos, pushee_pos, 1), 360)
                                
                                
                                #prev_pos_velo = ss_take_velocity_direction([], velo_theta_thresh[0, push_id[0, push_count]].copy(), pusher_pushee_angle)
                                prev_pos_velo = ss_take_velocity_direction([], 0.0, pusher_pushee_angle)
                                
                                prev_pos_velo = normalize(prev_pos_velo, axis=0)
                                prev_pos_velo = max_speed[0, snp]*prev_pos_velo
                                max_speed[0, push_id[1, push_count]] = 0
                                if((Temp_dist-2*object_radius-pushing_contact_distance) < pusher_speed_during_pushing):
                                    new_max_speed = Temp_dist-2*object_radius-pushing_contact_distance
                                else:
                                    new_max_speed = pusher_speed_during_pushing
                                                                    
                            else:#end pusher movement
                                pushing_start_flag = 0
                                pushing_halt_flag = 1
                                pushing_delay_count = 0
                                new_max_speed = 0.0
                     
                        elif(pushing_halt_flag):#pushing halt
                            if(pushing_delay_count < pushing_delay):
                                max_speed[0, push_id[0, push_count]] = 0
                                max_speed[0, push_id[1, push_count]] = 0
                                pushing_delay_count += 1
                            else:#start pushee movement
                                pusher_halt_count = 0
                                pushing_start_flag = 0
                                pushing_halt_flag = 0
                                new_max_speed = 0.0#1
                                max_speed[0, push_id[0, push_count]] = new_max_speed
                                push_frame_info[2, push_count] = snt
                                
                                pushing_end_flag = 1
        
                                pusher_pos = np.reshape(pos[:,push_id[0, push_count],snt-1], (pos.shape[0], 1)).copy()
                                pushee_pos = np.reshape(pos[:,push_id[1, push_count],snt-1], (pos.shape[0], 1)).copy()
                                pusher_pushee_angle = np.mod(ss_anglefrom2points(pusher_pos, pushee_pos, 1), 360)
                                pushee_velo = avegrate_max_speed*ss_take_velocity_direction([], pushing_subtlety, pusher_pushee_angle, [])
                                
                                pos_velo[:,push_id[1, push_count],snt-1] = np.reshape(pushee_velo.copy(),(pos_velo.shape[0],))#pos_velo[:,push_id[0, push_count],snt-1].copy()
                                #print('TTTT-2:{}\n' .format(pos_velo[:,push_id[1, push_count],snt-1]))
                                max_speed[0, push_id[1, push_count]] = avegrate_max_speed - new_max_speed
                                                            
                        else:
                            
                            new_max_speed = max_speed[0, push_id[0, push_count]].copy()
                            new_max_speed = min(new_max_speed+0.5, avegrate_max_speed)
                            max_speed[0, push_id[0, push_count]] = new_max_speed.copy()
                            if(snt < push_frame_info[2, push_count] + pusher_halt):
                                prev_pos_velo =  np.reshape(pos_velo[:,push_id[1, push_count],snt-1], (pos_velo.shape[0], 1)).copy()
                            if((pusher_halt_count < pusher_halt)):
                                max_speed[0, push_id[0, push_count]] = 0
                                pusher_halt_count += 1
                    #FOR PUSHEE MOTION
                    elif(snp == push_id[1, push_count]):
                        if(pushing_end_flag):
                            new_max_speed = max_speed[0, push_id[1, push_count]]
                            max_speed[0, push_id[1, push_count]] = min(new_max_speed+1, avegrate_max_speed)
                            prev_pos_velo =  np.reshape(pos_velo[:,push_id[1, push_count],snt-1], (pos_velo.shape[0], 1)).copy()
                            prev_pos_velo_all = np.reshape(pos_velo[:,:,snt-1], (pos_velo.shape[0], pos_velo.shape[1])).copy()
                            
                        else:
                            if((pushing_halt_flag>0) and (pushing_delay < 1)):
                                new_max_speed = max_speed[0, push_id[1, push_count]]
                                max_speed[0, push_id[1, push_count]] = min(new_max_speed+1, avegrate_max_speed)
                                prev_pos_velo =  np.reshape(pos_velo[:,push_id[1, push_count],snt-1], (pos_velo.shape[0], 1)).copy()
                                prev_pos_velo_all = np.reshape(pos_velo[:,:,snt-1], (pos_velo.shape[0], pos_velo.shape[1])).copy()
                            else:
                                new_max_speed = 0
                    else:
                        if((pushing_start_flag>0) or (pushing_halt_flag>0)):     
                            new_max_speed = 0

                        elif((pushing_start_flag==0) and (pushing_halt_flag==0) and (pushing_end_flag>0.0) and (snt<push_frame_info[2, push_count]+pusher_halt)):
                            new_max_speed = 0
                            
                        else:
                            new_max_speed = avegrate_max_speed
                            if(snt == push_frame_info[2, push_count]+pusher_halt):
                                prev_pos_velo = np.reshape(int_pos_velo[:,snp], (pos_velo.shape[0], 1)).copy()
                            
                if(snt == push_frame_info[1, push_count]):
                    pushing_end_flag = 0
                    pushing_start_flag = 1
                    push_count += 1
            
            if((pushing_end_flag>0.0) and (snp == push_id[1, push_count]) ):
                Temp_boundary_steer_dirc, Temp_boundary_dirc, Temp_boundary_des_dirc, Temp_boundary_des_dirc1 = ss_border_neighbour_constraints(prev_pos, prev_pos_velo, row, column, boundary_dx_row, boundary_dx_column, prev_pos_all, prev_pos_velo_all, 0.0, max_speed[0, snp], combined_max_steer_force)#neighbour_disjoint_threshold=0.0
            else:
                Temp_boundary_steer_dirc, Temp_boundary_dirc, Temp_boundary_des_dirc, Temp_boundary_des_dirc1 = ss_border_neighbour_constraints(prev_pos, prev_pos_velo, row, column, boundary_dx_row, boundary_dx_column, prev_pos_all, prev_pos_velo_all, neighbour_disjoint_threshold, max_speed[0, snp], combined_max_steer_force)
            Temp_neighbour_steer_dirc = np.zeros((2,1))

            new_pos_accl = Temp_neighbour_steer_dirc + Temp_boundary_steer_dirc
            new_pos_velo = prev_pos_velo + new_pos_accl
            new_pos_velo = normalize(new_pos_velo, axis=0)
            new_pos_velo = new_max_speed*new_pos_velo
            
            new_pos = np.round(prev_pos + new_pos_velo)
            new_pos = ss_check_bounday(new_pos, row, column, dx_row/2., dx_column/2.)
            
            pos[:,snp,snt] = np.reshape(new_pos, (new_pos.shape[0],)).copy()
            pos_velo[:,snp,snt] = np.reshape(new_pos_velo, (new_pos_velo.shape[0],)).copy()

    return pos, pos_velo, push_frame_info, push_id
      
   

