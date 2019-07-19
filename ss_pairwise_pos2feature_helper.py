###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 140817/210817/031017/050918
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
#%reset -f
import numpy as np
import matplotlib.pyplot as plt
##############################################################################################

import os
import sys

ss_lib_path = 'SS_PYLIBS/'
sys.path.insert(0, ss_lib_path)
#---------------------------------------------------------------------------------------------
from ss_computation import *
##############################################################################################

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2
##############################################################################################

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 140817
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_data_scaled_prepros(data, disp_flag=0):
    
    # For visualization 
    if(disp_flag):
        plt.subplot(2, 1, 1)
        plt.title('Original')
        plt.subplot(2, 1, 2)
        plt.title('Scaled')

        plt.subplot(2,1,1)
        for i in range(data.shape[1]):
            x = data[0,i,:]
            y = data[1,i,:]
            plt.plot(x, y, '+', label='p'+str(i))
        
        
    if(data.ndim == 3):
        # Find the bounday points in each dimension
        x_min = data[0,:,:].min()
        x_max = data[0,:,:].max()
        y_min = data[1,:,:].min()
        y_max = data[1,:,:].max()
#         print(x_min, x_max, y_min, y_max)
        origin_x = (x_min+x_max)//2
        origin_y = (y_min+y_max)//2
        data[0,:,:] -= origin_x # shift the origin x-position
        data[1,:,:] -= origin_y# shift the origin y-position
#         print(origin_x, origin_y)
        
        scale_factor = max([x_max-x_min, y_max-y_min])/2 + 1 # scale +1 confirm [-1, 1]
        data /= scale_factor 
        
        # For visualization 
        if(disp_flag):
            plt.subplot(2,1,2)
            for i in range(data.shape[1]):
                x = data[0,i,:]
                y = data[1,i,:]
                plt.plot(x, y, '+', label='p'+str(i))
            plt.legend(loc='upper center', ncol=4)
            plt.gcf().set_size_inches(15, 15)
            plt.show()
    
    else:
        raise ValueError('Please check the data dimensions (take 3-d with format: 2*num_points*num_frames)')

    return(data)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 120817/150817/031017
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_position_velocity_data_prepros(data):
    
    if(data.ndim == 3):
        num_points = data.shape[1]
        Temp_data = [None]*num_points
        for sno in range(num_points):
            Temp_data[sno] = data[:,sno, :].T
            #Temp_data[sno] = np.hstack([Temp_data[sno][1:], Temp_data[sno][1:] - Temp_data[sno][:-1]])
            
            Temp_data_velo = Temp_data[sno][1:] - Temp_data[sno][:-1]
            Temp_data_velo_magnitude = np.sqrt(np.sum(Temp_data_velo**2, axis=1))
            max_Temp_data_velo_magnitude = Temp_data_velo_magnitude.max()
            Temp_data_velo /= ss_denominator_check(max_Temp_data_velo_magnitude)
            Temp_data[sno] = np.hstack([Temp_data[sno][1:], Temp_data_velo])

        data = np.hstack(Temp_data)
    else:
        raise ValueError('Please check the data dimensions (take 3-d with format: 2*num_points*num_frames)')

    return data

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 120817/150817/210817
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################        
def ss_nn_chasing_data(data, chasing_id=[1,0], frames_per_action=30, stride=10):
    
    num_frames = data.shape[0]
    
    if(frames_per_action<=num_frames):
        
        num_points = data.shape[1]//4
        Temp_idx = range(frames_per_action, num_frames+1, stride)
        num_pairs = num_points*num_points - num_points # exclude self pairs like (1,1), (2,2),...
        total_data_points = len(Temp_idx)
        data_dimension = 8*frames_per_action
        new_data = {}#[None]*num_pairs
        new_labels = {}#[None]*num_pairs
        count_1 = 0
        for i in range(num_points):
            xy_vxvy_i = data[:,(4*i):(4*i+4)]
            for j in range(num_points):

                if(i==j):
                    continue
                else:
                    Temp_data = [None]*total_data_points
                    xy_vxvy_j = data[:,(4*j):(4*j+4)]
                    for snf in range(total_data_points):
                        X_i = xy_vxvy_i[(Temp_idx[snf]-frames_per_action):Temp_idx[snf],:]
                        X_j = xy_vxvy_j[(Temp_idx[snf]-frames_per_action):Temp_idx[snf],:]
                        Temp_data[snf] = np.reshape(np.hstack([X_i, X_j]), (-1,1))
                    new_data[(i,j)] = np.hstack(Temp_data)
                    if((i==chasing_id[0]) and (j==chasing_id[1])):
                        new_labels[(i,j)] = np.ones(total_data_points)
                    else:
                        new_labels[(i,j)] = np.zeros(total_data_points)                   
    else:
        raise ValueError('Requested # frames ({}) > input video # frames ({})' .format(frames_per_action, num_frames))
                        
    return(new_data, new_labels, Temp_idx)
   
###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 120817/150817/210817
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################    
def ss_nn_pushing_data(data, pushing_id, pushing_frame_info, frames_per_action=30, stride=10):
    
    num_frames = data.shape[0]
    
    if(frames_per_action<=num_frames):
        
        num_points = data.shape[1]//4
        num_pushing = pushing_id.shape[1]
        
        push_thresh_right = 5
        push_thresh_left = 10#frames_per_action-2*push_thresh_right#int(frames_per_action//2.)
        
        Temp_idx = range(frames_per_action, num_frames+1, stride)
        
        num_pairs = num_points*num_points - num_points # exclude self pairs like (1,1), (2,2),...
        total_data_points = len(Temp_idx)
        data_dimension = 8*frames_per_action
        new_data = {}#[None]*num_pairs
        new_labels = {}#[None]*num_pairs
       
        for i in range(num_points):
            xy_vxvy_i = data[:,(4*i):(4*i+4)]
            for j in range(num_points):
               
                if(i==j):
                    continue
                else:
                    Temp_data = [None]*total_data_points
                    Temp_label = np.zeros(total_data_points)
                    xy_vxvy_j = data[:,(4*j):(4*j+4)]
                    for snf in range(total_data_points):
                        X_i = xy_vxvy_i[(Temp_idx[snf]-frames_per_action):Temp_idx[snf],:]
                        X_j = xy_vxvy_j[(Temp_idx[snf]-frames_per_action):Temp_idx[snf],:]
                        Temp_data[snf] = np.reshape(np.hstack([X_i, X_j]), (-1,1))

                        for tsnp in range(num_pushing):
                            # Taking [exact_push_id-push_thresh_left exact_push_id-push_thresh_right] as 
                            # ground truth pusing labels
#                             if(((pushing_frame_info[2, tsnp]-push_thresh_left)>=(Temp_idx[snf]-frames_per_action))
#                                and ((pushing_frame_info[2, tsnp])<(Temp_idx[snf]-1))
#                                and ((pushing_frame_info[2, tsnp]+push_thresh_right)>(Temp_idx[snf]-1)) 
#                                and (i==pushing_id[0,tsnp]) and (j==pushing_id[1,tsnp])):
                            if(((pushing_frame_info[2, tsnp]-push_thresh_left)>=(Temp_idx[snf]-frames_per_action))
                               and ((pushing_frame_info[2, tsnp]+push_thresh_right)<(Temp_idx[snf])) 
                               and (i==pushing_id[0,tsnp]) and (j==pushing_id[1,tsnp])):
                                Temp_label[snf] = 2

                    new_data[(i,j)] = np.hstack(Temp_data)
                    new_labels[(i,j)] = Temp_label
    else:
        raise ValueError('Requested # frames ({}) > input video # frames ({})' .format(frames_per_action, num_frames))
                         
    return(new_data, new_labels, Temp_idx)
    
###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 120817/150817
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################        
def ss_chasing_data_2_vector(data, chasing_id=[0,1],  frames_per_action=30, stride=10, data_scale_flag=1): 
    
    if data_scale_flag:
        scaled_data = ss_data_scaled_prepros(data.copy())# Scaled the input data(shif origin and all are in [-1, 1])
    else:
        scaled_data = data.copy()
    prepros_data = ss_position_velocity_data_prepros(scaled_data.copy())# For position and velosity regression

    nn_data, nn_labels, nn_frame_idx = ss_nn_chasing_data(prepros_data.copy(), chasing_id, frames_per_action, stride)

    return(nn_data, nn_labels, nn_frame_idx)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 120817/150817
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_pushing_data_2_vector(data, pushing_id, pushing_frame_info,  frames_per_action=30, stride=10, data_scale_flag=1):   
    
    if data_scale_flag:
        scaled_data = ss_data_scaled_prepros(data.copy())# Scaled the input data(shif origin and all are in [-1, 1])
    else:
        scaled_data = data.copy()
    prepros_data = ss_position_velocity_data_prepros(scaled_data.copy())# For position and velosity regression

    nn_data, nn_labels, nn_frame_idx = ss_nn_pushing_data(prepros_data.copy(), pushing_id, pushing_frame_info, frames_per_action, stride)

    return(nn_data, nn_labels, nn_frame_idx)



