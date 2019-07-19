###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 150817/170817/210817/120917/180917/031017/141117/050918
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
# Some header and necessary function import
#%reset -f
import numpy as np
import math
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
#%load_ext autoreload
#%autoreload 2
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################

def fc_heuristic_averageVelocityMagnitude(pos, end_frame=5):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    
    velocity = pos[:,:,1:] - pos[:,:,:-1] # velocity
    
    velocity_magnitude = np.zeros((num_points, num_frames))
    velocity_magnitude[:,1:] = np.sqrt(np.sum(velocity*velocity, axis=0))# velocity magnitude
    
    # average velocity in last few frames
    velocity_magnitude_interval = np.zeros(velocity_magnitude.shape)
    for snf in range(num_frames):
        int_frame = snf - end_frame
        if int_frame<0:
            int_frame = 0
        velocity_magnitude_interval[:,snf] = velocity_magnitude[:,int_frame:snf+1].mean(axis=1)
    
    return velocity_magnitude_interval

def ss_Heuristic_RelativeDistance(pos):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    relative_distance = np.zeros((num_points, num_points, num_frames))
    
    for i in range(num_points):
        for j in range(num_points):
            Temp_diff = pos[:,i,:] - pos[:,j,:]
            relative_distance[i,j,:] = np.sqrt(np.sum(Temp_diff*Temp_diff, axis=0))

    return relative_distance
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AngleOfMotion(pos):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    angle_of_motion = np.zeros((num_points, num_frames))
    for snf in range(1, num_frames):
        for i in range(num_points):
            angle_of_motion[i, snf-1] = np.mod(ss_anglefrom2points(pos[:,i,snf-1], pos[:,i,snf])*(180/np.pi), 360)
                
    return angle_of_motion
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_RelativeAngleOfMotion(pos):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    relative_angle_of_motion = np.zeros((num_points, num_points, num_frames))
    for snf in range(1, num_frames):
        for i in range(num_points):
            Va = ss_theta2vector(np.mod(ss_anglefrom2points(pos[:,i,snf-1], pos[:,i,snf])*(180/np.pi), 360))
            for j in range(num_points):
                Vb = ss_theta2vector(np.mod(ss_anglefrom2points(pos[:,i,snf-1], pos[:,j,snf-1])*(180.0/np.pi), 360))
                
                # if i-th objcet is fixed then
#                 if(np.absolute(pos[:,i,snf-1] - pos[:,i,snf]).sum() == 0.0):
#                     Va = Vb.copy()
#                 relative_angle_of_motion[i, j, snf-1] = ss_cosine_anglefrom2vectors(Va, Vb)
                if((np.absolute(pos[:,i,snf-1] - pos[:,i,snf]).sum() == 0.0) and (snf-1 >=0)):
                    relative_angle_of_motion[i, j, snf-1] = relative_angle_of_motion[i, j, snf-2]
                else:
                    relative_angle_of_motion[i, j, snf-1] = ss_cosine_anglefrom2vectors(Va, Vb)
                
    return relative_angle_of_motion
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_VelocityMagnitude(pos):
    
    velocity = pos[:,:,1:] - pos[:,:,:-1]
    velocity_magnitude = np.sqrt(np.sum(velocity*velocity, axis=0))
    
    return velocity_magnitude
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AvgVelocityMagnitude(pos, int_frame=0, end_frame=5):
    
    velocity = pos[:,:,int_frame+1:end_frame] - pos[:,:,int_frame:end_frame-1]
    velocity_magnitude = np.sqrt(np.sum(velocity*velocity, axis=0)).mean(axis=1)
    
    return velocity_magnitude
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AvgVelocityMagnitudeInterval(pos, interval_thgreh=30):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    velocity_magnitude = ss_Heuristic_VelocityMagnitude(pos)
    velocity_magnitude_interval = np.zeros(velocity_magnitude.shape)
    
    for snf in range(num_frames-1):
        int_frame = snf - interval_thgreh
        if int_frame<0:
            int_frame = 0
        velocity_magnitude_interval[:,snf] = velocity_magnitude[:,int_frame:snf+1].mean(axis=1)
    
    return velocity_magnitude_interval
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AvgVelocityBeforeAfter(pos, frame_diff_thresh=10):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    frame_thresh = int(frame_diff_thresh/2)
    int_frame = frame_thresh
    end_frame = num_frames-frame_thresh
    velocity_magnitude = ss_Heuristic_VelocityMagnitude(pos)

    before_velocity_diff = np.zeros((num_points, num_points, num_frames))
    after_velocity_diff = np.zeros((num_points, num_points, num_frames))

    for snf in range(int_frame, end_frame):
        for i in range(num_points):
            for j in range(num_points):
                avg_velo_before_touch_object_1 = velocity_magnitude[i,snf-frame_thresh:snf].mean()
                avg_velo_after_touch_object_1 = velocity_magnitude[i,snf+1:snf+frame_thresh].mean()
                avg_velo_before_touch_object_2 = velocity_magnitude[j,snf-frame_thresh:snf].mean()
                avg_velo_after_touch_object_2 = velocity_magnitude[j,snf+1:snf+frame_thresh].mean()
                before_velocity_diff[i,j,snf] = avg_velo_before_touch_object_1 - avg_velo_before_touch_object_2
                after_velocity_diff[i,j,snf] = avg_velo_after_touch_object_2 - avg_velo_after_touch_object_1
              
    return before_velocity_diff, after_velocity_diff
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AccelerationMagnitude(pos):
    
    velocity = pos[:,:,1:] - pos[:,:,:-1]
    acceleration = velocity[:,:,1:] - velocity[:,:,:-1]
    acceleration_magnitude = np.sqrt(np.sum(acceleration*acceleration, axis=0))
    
    return acceleration_magnitude
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 2210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_AvgAccelerationMagnitudeInterval(pos, interval_thgreh=30):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    acceleration_magnitude = ss_Heuristic_AccelerationMagnitude(pos)
    acceleration_magnitude_interval = np.zeros(acceleration_magnitude.shape)
    
    for snf in range(num_frames-2):
        int_frame = snf - interval_thgreh
        if int_frame<0:
            int_frame = 0
        acceleration_magnitude_interval[:,snf] = acceleration_magnitude[:,int_frame:snf+1].mean(axis=1)
    
    return acceleration_magnitude_interval
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_DelayCount(pos, stop_thresh=0.0):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    velocity_magnitude = ss_Heuristic_VelocityMagnitude(pos)

    velocity_magnitude = velocity_magnitude <= stop_thresh
    delay_info = np.zeros((num_points, num_frames))
    for i in range(num_points):
        delay_info[i, 0] = 0
        for snf in range(1, num_frames-1):
            if velocity_magnitude[i, snf]:
                delay_info[i, snf] = delay_info[i, snf-1] + 1.

    return delay_info
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_Heuristic_DelayCountInterval(pos, stop_thresh=0.0, interval_thgreh=30):
    
    num_points, num_frames = pos.shape[1], pos.shape[2]
    velocity_magnitude = ss_Heuristic_VelocityMagnitude(pos)

    velocity_magnitude = velocity_magnitude <= stop_thresh
    delay_info = ss_Heuristic_DelayCount(pos, stop_thresh=0.0)
    interval_delay_info = np.zeros((num_points, num_frames))
    interval_thgreh = interval_thgreh//2
    for i in range(num_points):
        for snf in range(num_frames-1):
            if velocity_magnitude[i, snf]:
                int_frame = snf - interval_thgreh
                if int_frame<0:
                    int_frame = 0
                end_frame = snf + interval_thgreh
                if end_frame>num_frames:
                    end_frame = num_frames
                interval_delay_info[i, snf] = delay_info[i, int_frame:end_frame].max()
                        
    return delay_info, interval_delay_info
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_chasing_detection(pos, action_start_frame_id=10, heuristic_add_factor=0.99, class_id_threshold=40):

    #calculate relative angle of motions(relative_angle_of_motion)
    relative_angle_of_motion = ss_Heuristic_RelativeAngleOfMotion(pos)
    
    #cumulative calculation of relative_angle_of_motion over number of frames
    heuristic_value = relative_angle_of_motion.copy()
    num_frames = relative_angle_of_motion.shape[2]
    
    for stnf in range(action_start_frame_id, num_frames):
        heuristic_value[:,:,stnf]  = heuristic_add_factor*heuristic_value[:,:,stnf-1] + (1-heuristic_add_factor)*relative_angle_of_motion[:,:,stnf] 
    Temp_heuristic_value = heuristic_value[:,:,-1].copy()
    max_value = Temp_heuristic_value.max() + 10.
    np.fill_diagonal(Temp_heuristic_value, max_value)
    min_value = Temp_heuristic_value.min()
    est_chasing_idx = np.array(np.where(Temp_heuristic_value == min_value))#Find the minimum relative_angle_of_motion 

    # scale the heuristic values
    for i in range(heuristic_value.shape[0]):
        heuristic_value[i,i,:] = max_value
    heuristic_value -= min_value
    heuristic_value /= heuristic_value.max()
    heuristic_value *= 8.
    heuristic_value = np.exp(-heuristic_value)
    
    #for existance of chasing
    relative_distance = ss_Heuristic_RelativeDistance(pos)
    relative_distance = relative_distance[:,:,action_start_frame_id:-1]
    V_relative_distance = relative_distance.std(axis=2)
    total_std = V_relative_distance[np.eye(V_relative_distance.shape[0])<1].std()
    if total_std < class_id_threshold:
        est_class_ids = 1
    else:
        est_class_ids = 0
    
                    
    return est_class_ids, est_chasing_idx, heuristic_value
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_chasing_match(org_chasing_info, est_chasing_info):

    num_chasing = org_chasing_info.shape[1]

    match_ids = np.zeros(num_chasing)
    count = 0
    for i in range(num_chasing):
        e_ids, _ = ss_matrix_compare(np.reshape(org_chasing_info[:,i], (2, 1)) , est_chasing_info[:,:], axis_flag=1)
        if e_ids.size:
            match_ids[i] = e_ids
            count += 1
        else:
            match_ids[i] = -1

    match_ids = match_ids.astype(int)            
    
    return count, match_ids
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_pushing_squeeze(est_pushing_info, frame_diff_thresh=10):
    
    Temp_est_pushing_info = est_pushing_info.copy()
    Temp_count = 0
    Temp_est_pushing_info[:,Temp_count] = est_pushing_info[:,0] # final pushing list
    Temp_count += 1
    for i in range(1, est_pushing_info.shape[1]):
        e_ids, _ = ss_matrix_compare(np.reshape(Temp_est_pushing_info[:2,:Temp_count], (2, Temp_count)) , np.reshape(est_pushing_info[:2,i],(2, 1)), axis_flag=1)# find the match in the final pushing list
        #print(e_ids)
        if e_ids.size:
            Temp_frames = Temp_est_pushing_info[2,e_ids]
            if(np.abs(Temp_frames-est_pushing_info[2,i]).min() > frame_diff_thresh): # if the frame diff is not frame_diff_thresh(may vary) frames then it is same pushing else diff pushing 
                Temp_est_pushing_info[:,Temp_count] = est_pushing_info[:,i]
                Temp_count += 1
        else:
            Temp_est_pushing_info[:,Temp_count] = est_pushing_info[:,i]
            Temp_count += 1

    est_pushing_info = Temp_est_pushing_info[:,:Temp_count]
    
    return est_pushing_info
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_pushing_match(org_pushing_info, est_pushing_info, frame_diff_thresh=10):
    # frame_diff_thresh = 10
    num_pushing = org_pushing_info.shape[1]

    match_ids = np.zeros(num_pushing)
    count = 0
    for i in range(num_pushing):
        e_ids, _ = ss_matrix_compare(np.reshape(org_pushing_info[:2,i], (2, 1)) , est_pushing_info[:2,:], axis_flag=1)
        if e_ids.size:
            Temp_frame_diff = np.abs(org_pushing_info[2,i] - est_pushing_info[2, e_ids])
            e_ids = e_ids[np.where(Temp_frame_diff < frame_diff_thresh)]
            if e_ids.size:
                match_ids[i] = e_ids
                count += 1
        else:
            match_ids[i] = -1

    match_ids = match_ids.astype(int)            
    
    return count, match_ids
##############################################################################################

##############################################################################################
## FUNCTION:
## WRITER: SOUMITRA SAMANTA            DATE: 200717/070918/210918
## For bug and others mail me at soumitramath39@gmail.com
##--------------------------------------------------------------------------------------------
## INPUT:
## OUTPUT:
##--------------------------------------------------------------------------------------------
## EXAMPLE:
##
##############################################################################################
def ss_pushing_detection(pos, num_objects, object_radius=10, frame_diff_thresh=10, interval_thgreh=30):
    
    sf = 12.#6.#scale factor(6. best)
    
    #H1 = ss_Heuristic_AvgVelocityMagnitude(pos, 0, 10)
    H2 = ss_Heuristic_VelocityMagnitude(pos)
    H5 = ss_Heuristic_RelativeDistance(pos)

    num_points, num_frames= H5.shape[0], H5.shape[2]

    for i in range(num_points):
        H5[i, i,:] = 10000000 #put a large number here
    touch_info = np.where(H5<=2*object_radius+2)
#                         print(touch_info)
    num_touch_object = len(touch_info[0])
    est_touch_info = np.zeros((3, num_touch_object))
    Temp_count_1 = 0
    est_pushing_info = np.zeros((3, num_touch_object))
    Temp_count = 0

    if num_touch_object:
        for i in range(num_touch_object):
            avg_velo_before_touch_object_1 = H2[touch_info[0][i],touch_info[2][i]-frame_diff_thresh//2:touch_info[2][i]].mean()
            avg_velo_after_touch_object_1 = H2[touch_info[0][i],touch_info[2][i]+1:touch_info[2][i]+frame_diff_thresh//2].mean()
            avg_velo_before_touch_object_2 = H2[touch_info[1][i],touch_info[2][i]-frame_diff_thresh//2:touch_info[2][i]].mean()
            avg_velo_after_touch_object_2 = H2[touch_info[1][i],touch_info[2][i]+1:touch_info[2][i]+frame_diff_thresh//2].mean()

            if(avg_velo_before_touch_object_1 > avg_velo_before_touch_object_2+1):
                est_touch_info[0, Temp_count_1] = touch_info[0][i]
                est_touch_info[1, Temp_count_1] = touch_info[1][i]
                est_touch_info[2, Temp_count_1] = touch_info[2][i]
                Temp_count_1 += 1
            if((avg_velo_before_touch_object_1 > avg_velo_before_touch_object_2+1) and(avg_velo_after_touch_object_2 > avg_velo_after_touch_object_1+1)):# 1 is for discriminating touch in normal movement

                est_pushing_info[0, Temp_count] = touch_info[0][i]
                est_pushing_info[1, Temp_count] = touch_info[1][i]
                est_pushing_info[2, Temp_count] = touch_info[2][i]
                Temp_count += 1

    est_touch_info = est_touch_info[:,:Temp_count_1].astype(int)
    if est_touch_info.size:
        est_touch_info = ss_pushing_squeeze(est_touch_info)
        est_touch_info = est_touch_info[:,est_touch_info[2,:].argsort()]

    est_pushing_info = est_pushing_info[:,:Temp_count].astype(int)

    if est_pushing_info.size:
        est_class_ids = 1
        est_pushing_info = ss_pushing_squeeze(est_pushing_info)

        est_pushing_info = est_pushing_info[:,est_pushing_info[2,:].argsort()]

    else:
        est_class_ids = 0

    ##############################################################################################
#     #SOUMITRA
#     before_velocity_diff, after_velocity_diff = ss_Heuristic_AvgVelocityBeforeAfter(pos, interval_thgreh)
#     delay_info, interval_delay_info =  ss_Heuristic_DelayCountInterval(pos)
#     relative_distance = ss_Heuristic_RelativeDistance(pos)
#     relative_angle_of_motion = np.exp(-3.0*(ss_Heuristic_RelativeAngleOfMotion(pos)/180.))

#     Temp_delay_info = np.exp(-(delay_info.astype(float)))
#     Temp_interval_delay_info = np.exp(-(interval_delay_info+1))#/interval_thgreh))

#     Temp_relative_distance = relative_distance.copy()
#     Temp_relative_distance[xrange(num_points),xrange(num_points),:] = relative_distance.max()
#     #                         Temp_relative_distance = Temp_relative_distance/(2.*object_radius+1)
#     Temp_relative_distance = Temp_relative_distance-(2.*object_radius+2)
#     Temp_relative_distance = Temp_relative_distance*(Temp_relative_distance>0.0)
#     Temp_relative_distance = Temp_relative_distance.astype(float)/(sf+1.)
#     Temp_relative_distance = np.exp(-(Temp_relative_distance))

#     T_before_velocity_diff = before_velocity_diff.copy()#-1
#     #T_before_velocity_diff[T_before_velocity_diff<=0.0] = .00001
#     T_before_velocity_diff[T_before_velocity_diff<=0.5] = 2.5#.00001
#     T_before_velocity_diff = np.exp(-(1./ss_denominator_check(T_before_velocity_diff)))

#     T_after_velocity_diff = after_velocity_diff.copy()#-1
#     #T_after_velocity_diff[T_after_velocity_diff<=0.0] = .00001
#     T_after_velocity_diff[T_after_velocity_diff<=0.5] = 2.5#.00001
#     T_after_velocity_diff = np.exp(-(1./ss_denominator_check(T_after_velocity_diff)))

#     est_pushing_score = np.zeros(relative_distance.shape)
#     for i in range(num_objects):
#         for j in range(num_objects):
#             #est_pushing_score[i,j, :] = Temp_relative_distance[i,j, :]*Temp_delay_info[j,:]*T_before_velocity_diff[i,j, :]*T_after_velocity_diff[i,j, :]
#             est_pushing_score[i,j, :] = Temp_relative_distance[i,j, :]*relative_angle_of_motion[i,j,:]*Temp_delay_info[j,:]*T_before_velocity_diff[i,j, :]*T_after_velocity_diff[i,j, :]
            
    ##############################################################################################
    #FRANKLIN
    delay_info, interval_delay_info =  ss_Heuristic_DelayCountInterval(pos)
    relative_distance = ss_Heuristic_RelativeDistance(pos)
    relative_angle_of_motion = np.exp(-3.0*(ss_Heuristic_RelativeAngleOfMotion(pos)/180.))
    acceleration_magnitude_interval = ss_Heuristic_AvgAccelerationMagnitudeInterval(pos, 1)
    aveVelMag = fc_heuristic_averageVelocityMagnitude(pos, end_frame=20)# 60

    Temnp_acceleration_magnitude_interval = np.zeros((acceleration_magnitude_interval.shape[0], delay_info.shape[1]))
    Temnp_acceleration_magnitude_interval[:,Temnp_acceleration_magnitude_interval.shape[1]-acceleration_magnitude_interval.shape[1]:] = acceleration_magnitude_interval
    Temnp_acceleration_magnitude_interval[Temnp_acceleration_magnitude_interval<=0.0] = np.finfo(float).eps
    Temnp_acceleration_magnitude_interval = np.exp(-(1./Temnp_acceleration_magnitude_interval))

    Temnp_aveVelMag = aveVelMag.copy()
    Temnp_aveVelMag[Temnp_aveVelMag<=0.0] = np.finfo(float).eps
    Temnp_aveVelMag = np.exp(-(1./Temnp_aveVelMag))

    Temp_delay_info = np.exp(-(delay_info.astype(float)))
    Temp_interval_delay_info = np.exp(-(interval_delay_info+0.9))#/interval_thgreh))

    Temp_relative_distance = relative_distance.copy()
    Temp_relative_distance[xrange(num_points),xrange(num_points),:] = relative_distance.max()
    Temp_relative_distance = Temp_relative_distance-(2.*object_radius+2)
    Temp_relative_distance = Temp_relative_distance*(Temp_relative_distance>0.0)
    Temp_relative_distance = Temp_relative_distance.astype(float)/(sf+1.)
    Temp_relative_distance = np.exp(-(Temp_relative_distance))

    est_pushing_score = np.zeros(relative_distance.shape)
    for i in range(num_objects):
        for j in range(num_objects):
#                                 est_pushing_score[i,j, :] = Temp_relative_distance[i,j, :]*relative_angle_of_motion[i,j,:]*Temp_delay_info[j,:]*Temnp_acceleration_magnitude_interval[j,:]
#                                 est_pushing_score[i,j, :] = Temp_relative_distance[i,j, :]*relative_angle_of_motion[i,j,:]*Temnp_aveVelMag[j,:]*Temnp_acceleration_magnitude_interval[j,:]#221118
            est_pushing_score[i,j, :] = Temp_relative_distance[i,j, :]*relative_angle_of_motion[i,j,:]*Temnp_aveVelMag[i,:]*Temnp_acceleration_magnitude_interval[j,:]#231118

    ##############################################################################################
    
    
    for i in range(1, est_pushing_score.shape[2]):
        est_pushing_score[:,:,i] += est_pushing_score[:,:,i-1] 
    ##############################################################################################
    
    
        
    return est_touch_info, est_class_ids, est_pushing_info, [Temp_relative_distance, relative_angle_of_motion, Temnp_aveVelMag, Temnp_acceleration_magnitude_interval, Temp_delay_info, est_pushing_score]




