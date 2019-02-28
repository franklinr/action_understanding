
import numpy as np
from sets import Set
import copy
import math


from ss_computation import *
from ss_image_video import *
from ss_drawing import *
from ss_object_tracking import *


##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_ground_truth_wolf_sheep(obj_pos, obj_match_id, wolf_sheep_id, images, disp_flag = 0):
    
    wolf_sheep_points = ss_find_points_by_ids(obj_pos, obj_match_id, wolf_sheep_id)
    wolf_sheep_id = ss_arrayelements2string(np.array(wolf_sheep_id))

    num_frames = len(wolf_sheep_points)
    for nf in range(num_frames):
        images[nf] = ss_draw_textOnImage(images[nf],wolf_sheep_id, wolf_sheep_points[nf][[1,0],:])
        if(disp_flag):
            ss_images_show([images[nf]])
    return(wolf_sheep_points, images)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_match_trajectory(pos, traj_length, match_method = 'dist_var'):
    
    total_traj_length = len(pos)
#     traj_length = total_traj_length
#     if total_traj_length < 5:
    if total_traj_length < traj_length:
#         print('TRAJECTORY LENGTH IS TOO SHORT \n')
        return(np.array([]), np.array([]))
    else:
        Temp_pos = pos[-traj_length:]
        num_frames = len(Temp_pos)
        
        if match_method =='min_indv_ang_diff':
            
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points))
            for nf in range(num_frames-1, num_frames):
                for i in range(num_points):
                    Temp_1 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])*(180/np.pi), 360)
                    Va = ss_theta2vector(Temp_1)
                    for j in range(num_points):
                        Temp_2 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,j], Temp_pos[nf][:,j])*(180/np.pi), 360)
                        Vb = ss_theta2vector(Temp_2)
                        #Temp_3 = np.mod((Temp_2 - Temp_1), 360.0)
                        #Temp_dist[i,j] = np.fabs(Temp_3)
                        Temp_3 = ss_cosine_anglefrom2vectors(Va, Vb)
                        Temp_dist[i,j] = Temp_3
                       
            Temp_dist_sum = Temp_dist
            np.fill_diagonal(Temp_dist_sum, np.max(Temp_dist_sum)+2)
            min_idx = np.where(Temp_dist_sum == Temp_dist_sum.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
        
        elif match_method =='min_ang_diff':
            
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points))
            for nf in range(num_frames-1, num_frames):
                for i in range(num_points):
                    Temp_1 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])*(180.0/np.pi), 360)
                    Va = ss_theta2vector(Temp_1)
                    for j in range(num_points):
                        Temp_2 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf-1][:,j])*(180.0/np.pi), 360)
                        Vb = ss_theta2vector(Temp_2)
                        #Temp_3 = np.mod((Temp_2 - Temp_1), 360.0)
                        #Temp_3 = np.fabs(Temp_2 - Temp_1)
                        Temp_3 = ss_cosine_anglefrom2vectors(Va, Vb)
                        Temp_dist[i,j] = Temp_3
                       
            Temp_dist_sum = Temp_dist
            np.fill_diagonal(Temp_dist_sum, np.max(Temp_dist_sum)+2)
            min_idx = np.where(Temp_dist_sum == Temp_dist_sum.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
        
        elif match_method =='min_ang_diff_var':
            
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points, num_frames-1))
            for nf in range(1, num_frames):
                for i in range(num_points):
                    Temp_1 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])*(180.0/np.pi), 360)
                    Va = ss_theta2vector(Temp_1)
                    for j in range(num_points):
                        Temp_2 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf-1][:,j])*(180.0/np.pi), 360)
                        Vb = ss_theta2vector(Temp_2)
                        #Temp_3 = np.mod((Temp_2 - Temp_1), 360.0)
                        #Temp_dist[i,j] = np.fabs(Temp_3)
                        Temp_3 = ss_cosine_anglefrom2vectors(Va, Vb)
                        Temp_dist[i,j, nf-1] = Temp_3
                        
            Temp_dist = np.sum(Temp_dist, axis=2)           
            Temp_dist_sum = Temp_dist           
            np.fill_diagonal(Temp_dist_sum, np.max(Temp_dist_sum)+2)
            min_idx = np.where(Temp_dist_sum == Temp_dist_sum.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
        elif match_method =='traj_pos_similarity':
            num_points = Temp_pos[0].shape[1]
            Temp_dist = np.zeros((num_points, num_points, num_frames-1))
            #print(Temp_dist.shape)
            Temp_dist = np.zeros((Temp_pos[0].shape[1], Temp_pos[0].shape[1], num_frames-1))
            for nf in range(1, num_frames):
                Temp_1 = Temp_pos[nf] - Temp_pos[nf-1]
                #print(Temp_1.shape)
                Temp_dist[...,nf-1] = ss_euclidean_dist(Temp_1, Temp_1, 0)
                
            Temp_dist = np.sum(Temp_dist, axis=2)           
            Temp_dist_sum = Temp_dist
            np.fill_diagonal(Temp_dist_sum, np.max(Temp_dist_sum)+2)
            min_idx = np.where(Temp_dist_sum == Temp_dist_sum.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
            Temp_dist1 = np.zeros((num_points, num_points))
            for nf in range(num_frames-1, num_frames):
                for i in range(num_points):
                    Temp_1 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])*(180.0/np.pi), 360)
                    Va = ss_theta2vector(Temp_1)
                    for j in range(num_points):
                        Temp_2 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf-1][:,j])*(180.0/np.pi), 360)
                        Vb = ss_theta2vector(Temp_2)
                        #Temp_3 = np.mod((Temp_2 - Temp_1), 360.0)
                        #Temp_dist[i,j] = np.fabs(Temp_3)
                        Temp_3 = ss_cosine_anglefrom2vectors(Va, Vb)
                        Temp_dist1[i,j] = Temp_3
             
            Temp_dist_sum = Temp_dist1
            if(Temp_dist_sum[min_idx[0][0], min_idx[1][0]] > Temp_dist_sum[min_idx[1][0], min_idx[0][0]] ):
                match_pos = np.array([min_idx[1][0], min_idx[0][0]])
            
        elif match_method =='dist_var':        
            Temp_dist = np.zeros((Temp_pos[0].shape[1], Temp_pos[0].shape[1], num_frames))
            for nf in range(num_frames):
                Temp_dist[...,nf] = ss_euclidean_dist(Temp_pos[nf], Temp_pos[nf], 1)
             
            Temp_dist = np.var(Temp_dist, axis=2)
            Temp_dist_var = Temp_dist
            np.fill_diagonal(Temp_dist_var, np.max(Temp_dist_var)+2)
            min_idx = np.where(Temp_dist_var == Temp_dist_var.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
            num_points = Temp_pos[0].shape[1]
            Temp_dist1 = np.zeros((num_points, num_points))
            for nf in range(num_frames-1, num_frames):
                for i in range(num_points):
                    Temp_1 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])*(180.0/np.pi), 360)
                    Va = ss_theta2vector(Temp_1)
                    for j in range(num_points):
                        Temp_2 = np.mod(ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf-1][:,j])*(180.0/np.pi), 360)
                        Vb = ss_theta2vector(Temp_2)
                        #Temp_3 = np.mod((Temp_2 - Temp_1), 360.0)
                        Temp_3 = ss_cosine_anglefrom2vectors(Va, Vb)
                        Temp_dist1[i,j] = Temp_3
                       
            Temp_dist_sum = Temp_dist1
            if(Temp_dist_sum[min_idx[0][0], min_idx[1][0]] > Temp_dist_sum[min_idx[1][0], min_idx[0][0]] ):
                match_pos = np.array([min_idx[1][0], min_idx[0][0]])
            
            
        elif match_method =='traj_ang_similarity':
            
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points, num_frames-1))
            for nf in range(1, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf-1][:,j], Temp_pos[nf][:,j])
                        Temp_3 = Temp_2 - Temp_1
                        Temp_dist[i,j,nf-1] = np.fabs(Temp_3)
                       
            Temp_dist_sum = np.sum(Temp_dist, axis=2)
            np.fill_diagonal(Temp_dist_sum, np.max(Temp_dist_sum)+20)
            min_idx = np.where(Temp_dist_sum == Temp_dist_sum.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
        
        elif match_method =='indv_ang_diff_var':
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points, num_frames-1))
            for nf in range(1, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf-1][:,j], Temp_pos[nf][:,j])
                        Temp_3 = Temp_2 - Temp_1
                        Temp_dist[i,j,nf-1] = np.fabs(Temp_3)
                        
            Temp_dist_var = np.var(Temp_dist, axis=2)
            np.fill_diagonal(Temp_dist_var, np.max(Temp_dist_var)+20)
            min_idx = np.where(Temp_dist_var == Temp_dist_var.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
        elif match_method =='indv_ang_diff':
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_frames))
            Temp_dist_id = np.zeros((num_points, num_frames))
            for nf in range(1, 2):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    Temp_Temp_dist = np.zeros((num_points, 1))
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf-1][:,j], Temp_pos[nf][:,j])
                        Temp_3 = (Temp_2 - Temp_1)
                        Temp_Temp_dist[j] = Temp_dist[i,nf-1] + np.fabs(Temp_3)
                    Temp_Temp_dist[i] = Temp_Temp_dist.max() + 20
                    min_val, min_idx = Temp_Temp_dist.min(0), Temp_Temp_dist.argmin(0)
                    Temp_dist[i,nf] = Temp_dist[i,nf-1] + min_val
                    Temp_dist_id[i,nf] = min_idx                       
                        
            for nf in range(2, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    Temp_Temp_dist = np.zeros((num_points, 1))
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf-1][:,j], Temp_pos[nf][:,j])
                        Temp_3 = (Temp_2 - Temp_1)
                        Temp_Temp_dist[j] = Temp_dist[i,nf-1] + np.fabs(Temp_3)
                    Temp_Temp_dist[i] = Temp_Temp_dist.max() + 10
                    min_val, min_idx = Temp_Temp_dist.min(0), Temp_Temp_dist.argmin(0)
                    Temp_dist[i,nf] = min_val#Temp_dist[i,nf-1] + min_val
                    Temp_dist_id[i,nf] = min_idx
            min_idx = np.argmin(Temp_dist[:,-1])  
            match_pos = np.array([min_idx, Temp_dist_id[min_idx, -1]], int)
    
        elif match_method =='ang_diff_var':
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points, num_frames-1))
            for nf in range(1, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf][:,i], Temp_pos[nf][:,j])
                        Temp_3 = Temp_2 - Temp_1
                        Temp_dist[i,j,nf-1] = np.fabs(Temp_3)
                        
            Temp_dist_var = np.var(Temp_dist, axis=2)
            np.fill_diagonal(Temp_dist_var, np.max(Temp_dist_var)+20)
            min_idx = np.where(Temp_dist_var == Temp_dist_var.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
        elif match_method =='ang_diff':
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_frames))
            Temp_dist_id = np.zeros((num_points, num_frames))
            for nf in range(1, 2):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    Temp_Temp_dist = np.zeros((num_points, 1))
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf][:,i], Temp_pos[nf][:,j])
                        Temp_3 = (Temp_2 - Temp_1)
                        Temp_Temp_dist[j] = Temp_dist[i,nf-1] + np.fabs(Temp_3)
                    Temp_Temp_dist[i] = Temp_Temp_dist.max() + 20
                    min_val, min_idx = Temp_Temp_dist.min(0), Temp_Temp_dist.argmin(0)
                    Temp_dist[i,nf] = Temp_dist[i,nf-1] + min_val
                    Temp_dist_id[i,nf] = min_idx                       
                        
            for nf in range(2, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    Temp_Temp_dist = np.zeros((num_points, 1))
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf][:,i], Temp_pos[nf][:,j])
                        Temp_3 = (Temp_2 - Temp_1)
                        Temp_Temp_dist[j] = Temp_dist[i,nf-1] + np.fabs(Temp_3)
                    Temp_Temp_dist[i] = Temp_Temp_dist.max() + 10
                    min_val, min_idx = Temp_Temp_dist.min(0), Temp_Temp_dist.argmin(0)
                    Temp_dist[i,nf] = min_val#Temp_dist[i,nf-1] + min_val
                    Temp_dist_id[i,nf] = min_idx
            min_idx = np.argmin(Temp_dist[:,-1])  
            match_pos = np.array([min_idx, Temp_dist_id[min_idx, -1]], int)
            
        elif match_method =='dist_ang_var':
            
            num_points = Temp_pos[0].shape[1]
            
            Temp_dist = np.zeros((num_points, num_points, num_frames))
            for nf in range(num_frames):
                Temp_dist[...,nf] = ss_euclidean_dist(Temp_pos[nf], Temp_pos[nf], 1)
            Temp_dist_var = np.var(Temp_dist, axis=2)
            np.fill_diagonal(Temp_dist_var, np.max(Temp_dist_var)+2)
    
            Temp_angle = np.zeros((num_points, num_points, num_frames-1))
            for nf in range(1, num_frames):
                for i in range(num_points):
                    Temp_1 = ss_anglefrom2points(Temp_pos[nf-1][:,i], Temp_pos[nf][:,i])
                    for j in range(num_points):
                        Temp_2 = ss_anglefrom2points(Temp_pos[nf][:,i], Temp_pos[nf][:,j])
                        Temp_3 = Temp_2 - Temp_1
                        Temp_angle[i,j,nf-1] = np.fabs(Temp_3)
                        
            Temp_angle_var = np.var(Temp_angle, axis=2)           
            np.fill_diagonal(Temp_angle_var, np.max(Temp_angle_var)+2)
            lamda = 0.7
#             Temp_comb_var = Temp_dist_var + Temp_angle_var
            Temp_comb_var = lamda*Temp_dist_var + (1-lamda)*Temp_angle[:,:,-1]
            
            min_idx = np.where(Temp_comb_var == Temp_comb_var.min())
            match_pos = np.array([min_idx[0][0], min_idx[1][0]])
            
        else:
            print('ERROR:UNKNOWN MATCHING METHOD (%s) \n'%match_method)
            return(np.array([]), np.array([]))
            
#     print(match_pos)

    return(match_pos, Temp_dist)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 14-06-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_wolf_sheep_cost_image(wolf_sheep_cost, wolf_sheep_cost_image_size, wolf_sheep_id = [], wolf_sheep_color = np.reshape(np.array([[0],[255],[0]]),(1,1,3)), text_font_size = 0.4, text_color = np.array([[255],[0],[0]]), text_width = 1, verbose_flag = 0):#27-06-17
    
    row = wolf_sheep_cost_image_size[0]
    col = wolf_sheep_cost_image_size[1]
    image = np.zeros((row, col, 3))
    
    if wolf_sheep_cost.ndim == 2:
        if((wolf_sheep_cost.shape[0] <= row) & (wolf_sheep_cost.shape[1] <= col)):
            num_objects = wolf_sheep_cost.shape[0]
            tar_scale_range = [0.0, 255.0]
            stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range, 0)
            stretch_wolf_sheep_cost = stretch_wolf_sheep_cost.astype('int')
            cell_size_row = np.int(np.floor(row/wolf_sheep_cost.shape[0]))
            cell_size_col = np.int(np.floor(col/wolf_sheep_cost.shape[1]))
            Temp_id_row = np.arange(0, row, cell_size_row)
            Temp_id_col = np.arange(0, col, cell_size_col)
            for i in range(wolf_sheep_cost.shape[0]):
                for j in range(wolf_sheep_cost.shape[1]):
                    if len(wolf_sheep_id)>0:
                        if ((i == wolf_sheep_id[0]) & (j == wolf_sheep_id[1])):
                            image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = wolf_sheep_color*np.ones((cell_size_row, cell_size_col, 3))
                        else:
                            image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = (255.0-stretch_wolf_sheep_cost[i, j])*np.ones((cell_size_row, cell_size_col, 3))

                    else:
                        image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = (255.0-stretch_wolf_sheep_cost[i, j])*np.ones((cell_size_row, cell_size_col, 3))
                    Temp_i = Temp_id_row[i]+cell_size_row*0.7
                    Temp_j = Temp_id_col[j]                    
                    ss_draw_textOnImage(image, [str(stretch_wolf_sheep_cost[i, j])], np.array(([[Temp_j],[Temp_i]])), text_font_size, text_color, 1)
            
        else:
            if verbose_flag:
                print('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, wolf_sheep_cost.shape[0], wolf_sheep_cost.shape[1]))
    image = image.astype('uint8')
    
    return(image)

# def ss_wolf_sheep_cost_image(method_id, num_objects, wolf_sheep_cost, row, col, wolf_sheep_id = [], wolf_sheep_color = np.reshape(np.array([[0],[255],[0]]),(1,1,3)), text_font_size = 0.4, text_color = np.array([[255],[0],[0]]), text_width = 1, verbose_flag = 0):#26-06-17
    
#     image = np.zeros((row, col, 3))
    
#     if wolf_sheep_cost.ndim == 2:
#         if((wolf_sheep_cost.shape[0] <= row) & (wolf_sheep_cost.shape[1] <= col)):
#             num_objects = wolf_sheep_cost.shape[0]
#             tar_scale_range = [0.0, 255.0]
#             stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range, 0)
#             stretch_wolf_sheep_cost = stretch_wolf_sheep_cost.astype('int')
#             cell_size_row = np.int(np.floor(row/wolf_sheep_cost.shape[0]))
#             cell_size_col = np.int(np.floor(col/wolf_sheep_cost.shape[1]))
#             Temp_id_row = np.arange(0, row, cell_size_row)
#             Temp_id_col = np.arange(0, col, cell_size_col)
#             for i in range(wolf_sheep_cost.shape[0]):
#                 for j in range(wolf_sheep_cost.shape[1]):
#                     if len(wolf_sheep_id)>0:
#                         if ((i == wolf_sheep_id[0]) & (j == wolf_sheep_id[1])):
#                             image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = wolf_sheep_color*np.ones((cell_size_row, cell_size_col, 3))
#                         else:
#                             image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = (255.0-stretch_wolf_sheep_cost[i, j])*np.ones((cell_size_row, cell_size_col, 3))

#                     else:
#                         image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = (255.0-stretch_wolf_sheep_cost[i, j])*np.ones((cell_size_row, cell_size_col, 3))
#                     Temp_i = Temp_id_row[i]+cell_size_row*0.7
#                     Temp_j = Temp_id_col[j]                    
#                     ss_draw_textOnImage(image, [str(stretch_wolf_sheep_cost[i, j])], np.array(([[Temp_j],[Temp_i]])), text_font_size, text_color, 1)
            
#         else:
#             if verbose_flag:
#                 print('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, wolf_sheep_cost.shape[0], wolf_sheep_cost.shape[1]))
#     image = image.astype('uint8')
    
#     return(image)

# def ss_wolf_sheep_cost_image(method_id, num_objects, wolf_sheep_cost, row, col, text_font_size = 0.4, text_color = np.array([[255],[0],[0]]), text_width = 1, verbose_flag = 0):#21-06-17
    
#     image = np.zeros((row, col, 3))
    
#     if wolf_sheep_cost.ndim == 2:
#         if((wolf_sheep_cost.shape[0] <= row) & (wolf_sheep_cost.shape[1] <= col)):
#             num_objects = wolf_sheep_cost.shape[0]
#             tar_scale_range = [0.0, 255.0]
#             stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range, 0)
#             stretch_wolf_sheep_cost = stretch_wolf_sheep_cost.astype('int')
#             cell_size_row = np.int(np.floor(row/wolf_sheep_cost.shape[0]))
#             cell_size_col = np.int(np.floor(col/wolf_sheep_cost.shape[1]))
#             Temp_id_row = np.arange(0, row, cell_size_row)
#             Temp_id_col = np.arange(0, col, cell_size_col)
#             for i in range(wolf_sheep_cost.shape[0]):
#                 for j in range(wolf_sheep_cost.shape[1]):
#                     image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = stretch_wolf_sheep_cost[i, j]*np.ones((cell_size_row, cell_size_col, 3))
#                     Temp_i = Temp_id_row[i]+cell_size_row*0.7
#                     Temp_j = Temp_id_col[j]                    
#                     ss_draw_textOnImage(image, [str(stretch_wolf_sheep_cost[i, j])], np.array(([[Temp_j],[Temp_i]])), text_font_size, text_color, 1)
            
#         else:
#             if verbose_flag:
#                 print('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, wolf_sheep_cost.shape[0], wolf_sheep_cost.shape[1]))
#     image = image.astype('uint8')
    
#     return(image)

# def ss_wolf_sheep_cost_image(method_id, num_objects, wolf_sheep_cost, row, col, text_font_size = 0.4, text_color = np.array([[255],[0],[0]]), text_width = 1, obj_text_font_size = 0.5, obj_text_color = np.array([[0],[255],[0]]), obj_text_width = 1, meth_text_font_size = 0.5, meth_text_color = np.array([[0],[0],[0]]), meth_text_width = 1, verbose_flag = 0):#21-06-17
    
#     image = np.zeros((row, col, 3))
#     cell_size_row = np.int(np.floor(row/num_objects))
#     cell_size_col = np.int(np.floor(col/num_objects))
#     Temp_id_row = np.arange(0, row, cell_size_row)
#     Temp_id_col = np.arange(0, col, cell_size_col)
#     obj_id_image = np.zeros((row, cell_size_col, 3))# FOR OBJECT IDs
#     metbod_id_image = 255*np.ones((cell_size_row, col+cell_size_col, 3))#FOR METHOD IDs
#     for i in range(num_objects):
#         Temp_i = Temp_id_row[i]+cell_size_row*0.7
#         Temp_j = 0
#         ss_draw_textOnImage(obj_id_image, ['Ob'+str(i)], np.array(([[Temp_j],[Temp_i]])), obj_text_font_size, obj_text_color, obj_text_width)                
#     Temp_i = np.int(cell_size_row*0.8)
#     Temp_j = 0
#     metbod_id_image = ss_draw_textOnImage(metbod_id_image, method_id, np.array(([[Temp_j],[Temp_i]])), meth_text_font_size, meth_text_color, meth_text_width)
       
#     if wolf_sheep_cost.ndim == 2:
#         if((wolf_sheep_cost.shape[0] <= row) & (wolf_sheep_cost.shape[1] <= col)):
#             num_objects = wolf_sheep_cost.shape[0]
#             tar_scale_range = [0.0, 255.0]
#             stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range, 0)
#             stretch_wolf_sheep_cost = stretch_wolf_sheep_cost.astype('int')
#             for i in range(wolf_sheep_cost.shape[0]):
#                 for j in range(wolf_sheep_cost.shape[1]):
#                     image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = stretch_wolf_sheep_cost[i, j]*np.ones((cell_size_row, cell_size_col, 3))
#                     Temp_i = Temp_id_row[i]+cell_size_row*0.6
#                     Temp_j = Temp_id_col[j]                    
#                     ss_draw_textOnImage(image, [str(stretch_wolf_sheep_cost[i, j])], np.array(([[Temp_j],[Temp_i]])), text_font_size, text_color, 1)
            
#         else:
#             if verbose_flag:
#                 print('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, wolf_sheep_cost.shape[0], wolf_sheep_cost.shape[1]))
#     image = np.concatenate((obj_id_image, image), axis=1)
#     image = np.concatenate((image, metbod_id_image), axis=0)
#     image = image.astype('uint8')
    
#     return(image)

# def ss_wolf_sheep_cost_image(wolf_sheep_cost, row, col, verbose_flag = 0):
    
#     image = np.zeros((row, col, 3))
    
#     if wolf_sheep_cost.ndim == 2:
#         if((wolf_sheep_cost.shape[0] <= row) & (wolf_sheep_cost.shape[1] <= col)):
#             num_objects = wolf_sheep_cost.shape[0]
#             tar_scale_range = [0.0, 255.0]
#             stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range, 0)
#             cell_size_row = np.int(np.floor(row/wolf_sheep_cost.shape[0]))
#             cell_size_col = np.int(np.floor(col/wolf_sheep_cost.shape[1]))
#             Temp_id_row = np.arange(0, row, cell_size_row)
#             Temp_id_col = np.arange(0, col, cell_size_col)
#             for i in range(wolf_sheep_cost.shape[0]):
#                 for j in range(wolf_sheep_cost.shape[1]):
#                     image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = stretch_wolf_sheep_cost[i, j]*np.ones((cell_size_row, cell_size_col, 3))
#         else:
#             if verbose_flag:
#                 print('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, wolf_sheep_cost.shape[0], wolf_sheep_cost.shape[1]))

#     image = image.astype('uint8')
    
#     return(image)

# def ss_wolf_sheep_cost_image(wolf_sheep_cost, row, col, num_objects):
    
#     image = np.zeros((row, col, 3))
#     if len(wolf_sheep_cost) > 1:
#         wolf_sheep_cost = np.min(wolf_sheep_cost, axis=0)
#         tar_scale_range = [0.0, 255.0]
#         stretch_wolf_sheep_cost = ss_linear_scale_data(wolf_sheep_cost, tar_scale_range)
#         cell_size = np.int(np.floor(col/(num_objects+1.0)))
#         Temp_id = np.arange(0, col, cell_size)
#         for snd in range(num_objects):
#             image[:,Temp_id[snd]:Temp_id[snd]+cell_size,:] = stretch_wolf_sheep_cost[snd]*np.ones((row, cell_size, 3))
#         image[:,Temp_id[-1]:Temp_id[-1]+cell_size,:] = np.min(stretch_wolf_sheep_cost)*np.ones((row, cell_size, 3))
#     image = image.astype('uint8')
    
#     return(image)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_detection_wolf_sheep(track_pos, track_id, match_method, traj_length = 15, disp_flag = 0):#27-06-17
    
    num_frames = len(track_pos) 
    num_objects = track_pos[0].shape[1]
    int_pos = [None]*num_frames
    end_pos = [None]*num_frames
    est_wolf_sheep_ids = [None]*num_frames
    est_wolf_sheep_ids[0] = np.zeros(2)
    est_wolf_sheep_pos = [None]*num_frames
    est_wolf_sheep_pos[0] = np.zeros((2,2))
    est_wolf_sheep_cost_mat = [None]*num_frames
    est_wolf_sheep_cost_mat[0] = np.array([])
    
    for snf in range(1, num_frames):
        
        int_pos[snf] = track_pos[snf]
        end_pos[snf] = track_pos[snf]
                
        wolf_sheep_idx, Temp_wolf_sheep_cost = ss_match_trajectory(int_pos[1:snf+1], traj_length, match_method)
        est_wolf_sheep_cost_mat[snf] = Temp_wolf_sheep_cost
        if len(wolf_sheep_idx) > 1:
            est_wolf_sheep_ids[snf] = wolf_sheep_idx#np.array([track_id[wolf_sheep_idx[0]], track_id[wolf_sheep_idx[1]]])
            wolf_sheep_pos = end_pos[snf][:,wolf_sheep_idx]
            est_wolf_sheep_pos[snf] = wolf_sheep_pos
       
        else:
            est_wolf_sheep_ids[snf] = np.zeros(2)
            est_wolf_sheep_pos[snf] = np.zeros((2,2))

    return(est_wolf_sheep_ids, est_wolf_sheep_pos, est_wolf_sheep_cost_mat)

# def ss_detection_wolf_sheep(track_pos, track_id, match_method, images, traj_length = 15, wolf_sheep_cost_image_size = [40, 40], learn_rate_cuml = 0.8, disp_flag = 0):#26-06-17
    
#     num_frames = len(track_pos) 
#     num_objects = track_pos[0].shape[1]
#     int_pos = [None]*num_frames
#     end_pos = [None]*num_frames
#     est_wolf_sheep_ids = [None]*num_frames
#     est_wolf_sheep_ids[0] = np.zeros(2)
#     est_wolf_sheep_pos = [None]*num_frames
#     est_wolf_sheep_pos[0] = np.zeros((2,2))
    
    
#     Temp_color_code = ss_color_generation(len(track_pos[0])+50)
#     wolf_sheep_color_code = np.array([[255, 0, 0],[0, 255, 0]], int)
#     est_wolf_sheep_images = copy.deepcopy(images)
#     color_code = [None]*num_frames
#     text_pos = np.array([[10], [20]])
    
#     wolf_sheep_cost_image = [None]*num_frames
#     wolf_sheep_cuml_cost_image = [None]*num_frames
#     wolf_sheep_cuml_cost_mat = np.zeros((num_objects,num_objects))
#     wolf_sheep_cost_image_col = wolf_sheep_cost_image_size[0]#10*num_objects
#     wolf_sheep_cost_image_row = wolf_sheep_cost_image_size[1]#10*num_objects
#     wolf_sheep_cost_image[0] = ss_wolf_sheep_cost_image([match_method], num_objects, np.array([]), wolf_sheep_cost_image_row, wolf_sheep_cost_image_col)
#     wolf_sheep_cuml_cost_image[0] = ss_wolf_sheep_cost_image([match_method], num_objects, np.array([]), wolf_sheep_cost_image_row, wolf_sheep_cost_image_col)
    
#     for nf in range(1, num_frames):

#         color_code[nf] = Temp_color_code[:,track_id]
        
#         int_pos[nf] = track_pos[nf-1]
#         end_pos[nf] = track_pos[nf]
#         T_T = copy.deepcopy(images[nf])
                
#         wolf_sheep_idx, Temp_wolf_sheep_cost = ss_match_trajectory(int_pos[1:nf], traj_length, match_method)
#         cuml_wolf_sheep_idx = []
#         if Temp_wolf_sheep_cost.ndim == 2:
#             wolf_sheep_cuml_cost_mat = learn_rate_cuml*wolf_sheep_cuml_cost_mat + (1-learn_rate_cuml)*Temp_wolf_sheep_cost
#             TT_min_idx = np.where(wolf_sheep_cuml_cost_mat == wolf_sheep_cuml_cost_mat.min())
#             cuml_wolf_sheep_idx = np.array([TT_min_idx[0][0], TT_min_idx[1][0]])
#         wolf_sheep_cost_image[nf] = ss_wolf_sheep_cost_image([match_method], num_objects, Temp_wolf_sheep_cost, wolf_sheep_cost_image_row, wolf_sheep_cost_image_col, wolf_sheep_idx)
#         wolf_sheep_cuml_cost_image[nf] = ss_wolf_sheep_cost_image([match_method], num_objects, wolf_sheep_cuml_cost_mat, wolf_sheep_cost_image_row, wolf_sheep_cost_image_col, cuml_wolf_sheep_idx)
#         if len(wolf_sheep_idx) > 1:
#             est_wolf_sheep_ids[nf] = wolf_sheep_idx#np.array([track_id[wolf_sheep_idx[0]], track_id[wolf_sheep_idx[1]]])
#             wolf_sheep_pos = end_pos[nf][:,wolf_sheep_idx]
#             est_wolf_sheep_pos[nf] = wolf_sheep_pos
#             wolf_sheep_radius = 10*np.ones(wolf_sheep_pos.shape[1], np.int)
#             wolf_sheep_circumference_width = 5*np.ones(wolf_sheep_pos.shape[1], np.int)
#             for t_nf in range(nf-traj_length+1, nf+1):
#                 color_code[t_nf][:,wolf_sheep_idx] =  np.transpose(wolf_sheep_color_code)
#             T_T = ss_draw_circlesOnImage(T_T, wolf_sheep_pos[[1, 0],:], wolf_sheep_radius, wolf_sheep_color_code, wolf_sheep_circumference_width)
            
#         else:
#             est_wolf_sheep_ids[nf] = np.zeros(2)
#             est_wolf_sheep_pos[nf] = np.zeros((2,2))
#         for t_nf in range(1, nf):
#             T_T = ss_draw_lineOnImage(T_T, int_pos[t_nf][[1, 0],:], end_pos[t_nf][[1, 0],:], color_code[t_nf])
#         T_T = ss_draw_textOnImage(T_T, [match_method], text_pos, 1, np.array([[255], [0], [0]]))

#         est_wolf_sheep_images[nf] = T_T
#         if disp_flag:
#             ss_images_show([est_wolf_sheep_images[nf]])
#     return(est_wolf_sheep_ids, est_wolf_sheep_pos, est_wolf_sheep_images, wolf_sheep_cost_image, wolf_sheep_cuml_cost_image)


##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_detection_wolf_sheep_all_frames(est_wolf_sheep_ids, num_points):
    
    match_matrix = np.zeros((num_points, num_points))
    num_frames = len(est_wolf_sheep_ids)
    for nf in range(num_frames):
        Temp_id = np.array(est_wolf_sheep_ids[nf], int)
        Temp = match_matrix[int(est_wolf_sheep_ids[nf][0]), int(est_wolf_sheep_ids[nf][1])]
#         print(Temp)
        match_matrix[int(est_wolf_sheep_ids[nf][0]), int(est_wolf_sheep_ids[nf][1])] = Temp + 1
#     print(match_matrix)
    np.fill_diagonal(match_matrix, np.min(match_matrix))
    
    max_idx = np.where(match_matrix == match_matrix.max())
    match_pos = np.array([max_idx[0][0], max_idx[1][0]])
    
    return(match_pos, match_matrix)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_wolf_track_accuracy(true_pos, true_wolf_id, est_pos, est_wolf_id, dist_thersh = 0):
    num_frames = len(true_pos)
    count = np.zeros((1,num_frames))
    point_dist = [None]*num_frames
    for nf in range(num_frames):
        
        Temp_1 = np.sqrt(np.sum((true_pos[nf][:, true_wolf_id] - est_pos[nf][:, est_wolf_id])*(true_pos[nf][:, true_wolf_id] - est_pos[nf][:, est_wolf_id]), axis=0))
        point_dist[nf] = Temp_1
        
        if(Temp_1 <= dist_thersh):
            count[0,nf] = 1
    accuracy = 100*(np.float64(np.sum(count))/num_frames)
    return(accuracy, point_dist)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 21-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_wolf_sheep_track_accuracy(true_pos, est_pos, dist_thersh = 0):
    num_frames = len(true_pos)
    count = np.zeros((1,num_frames))
    point_dist = [None]*num_frames
    for nf in range(num_frames):
        
        Temp_1 = np.sqrt(np.sum((true_pos[nf] - est_pos[nf])*(true_pos[nf] - est_pos[nf]), axis=0))
        Temp_2 = np.sqrt(np.sum((true_pos[nf] - est_pos[nf][:,[1,0]])*(true_pos[nf] - est_pos[nf][:,[1,0]]), axis=0))
        point_dist[nf] = np.zeros((2,len(Temp_1)))
        point_dist[nf][0,:] = Temp_1
        point_dist[nf][1,:] = Temp_2
        
        Temp_1[Temp_1 <= dist_thersh] = 0
        Temp_2[Temp_2 <= dist_thersh] = 0
        
        if((np.sum(Temp_1) == 0.0) or (np.sum(Temp_2) == 0.0)):
            count[0,nf] = 1
    accuracy = 100*(np.float64(np.sum(count))/num_frames)
    return(accuracy, point_dist)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_find_push(required_points):
    
    num_frames = len(required_points)
    moving_points_id = [None]*num_frames

    pushed_flag = 0
    pusher_flag = 0
    pushed_id = -1
    pusher_id = -1
    push_info = [None]*num_frames
    push_count = 0
    for snf in range(2, num_frames):
        moving_points_id[snf] = ss_find_moving_objects(required_points, snf, 2.2)
        if ((len(moving_points_id[snf]) == 1) & (pusher_id < 0)): 
            pusher_flag = 1
            pusher_id = moving_points_id[snf][0]

        if (pusher_flag == 1):
            num_moving_objects = len(moving_points_id[snf])
            if num_moving_objects > 2:
                push_info[push_count] = [pusher_id, pushed_id]
                push_count = push_count + 1

                pusher_flag = 0
                pusher_id = -1

            else:
                for snmb in range(num_moving_objects):
                    if moving_points_id[snf][snmb] != pusher_id:
                        pushed_id = moving_points_id[snf][snmb]
    push_info = push_info[:push_count]    

    return(push_info)
