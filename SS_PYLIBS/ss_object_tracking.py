
import numpy as np
import copy
from sets import Set

from ss_computation import *
from ss_image_video import *
from ss_drawing import *
from ss_kalman_filter import *

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius):
    
    radii = np.zeros((1,2))
    centers = np.zeros((2,2))
    r = axis_length
    theta = -axis_orientation + np.pi/2
    x = (r/2.0 - reguired_objects_radius)*np.cos(theta)
    y = (r/2.0 - reguired_objects_radius)*np.sin(theta)
    Temp1 = np.array([x, -y])
    C1 = centroid + Temp1
    radii[0, 0] = reguired_objects_radius
    centers[:,0] = C1

    x1 = (r/2.0 - reguired_objects_radius)*np.cos(theta+np.pi)
    y1 = (r/2.0 - reguired_objects_radius)*np.sin(theta+np.pi)
    Temp2 = np.array([x1, -y1])
    C2 = centroid + Temp2
    radii[0, 1] = reguired_objects_radius
    centers[:,1] = C2
    
    return(centers, radii)


##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_joint_object_separation(stats, connected_object_id, num_reguired_objects, reguired_objects_radius):

    check_flag = 0
    if(check_flag):
        print('Checking "ss_joint_object_separation" \n')
    
    num_connected_object = len(connected_object_id)
    radii = np.zeros((1, np.sum(num_reguired_objects)))
    centers = np.zeros((2, np.sum(num_reguired_objects)))
    int_idc = 0
    for nco in range(num_connected_object):
        if num_reguired_objects[nco] == 2:
            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].major_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

        elif num_reguired_objects[nco] == 3:
            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].major_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

            end_idc = int_idc + 1
            centers[:,int_idc:end_idc] = np.reshape(centroid,(2, 1))
            radii[0, int_idc:end_idc] = reguired_objects_radius
            int_idc = end_idc

        elif num_reguired_objects[nco] == 4:
            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].major_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].minor_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation + np.pi/2
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

        else:
            print('WORNING:WE ARE NOT CONSIDERING (%d) OR MORE OBJECTS CONNECTED TOGETHER (<=4) \n' %num_reguired_objects[nco])
            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].major_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

            centroid = stats[connected_object_id[nco]].centroid
            axis_length = stats[connected_object_id[nco]].minor_axis_length
            axis_orientation = stats[connected_object_id[nco]].orientation + np.pi/2
            Temp_centers, Temp_radii = ss_two_object_circle(centroid, axis_length, axis_orientation, reguired_objects_radius)
            end_idc = int_idc + Temp_centers.shape[1]
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = Temp_radii
            int_idc = end_idc

    ##############################PICK OTHERS POINT RANDOMLY##############################
            temp_pixel_list = np.transpose(stats[connected_object_id[nco]].coords)
            Temp_centers = centers[:,int_idc-4:int_idc]
            Temp_dist = ss_euclidean_dist(Temp_centers, temp_pixel_list, 1);

            temp_pixel_id = np.array(range(Temp_dist.shape[1]))
            temp_pixel_id1 = Set([])

            for T_C in range(4):
                Temp_dist1 = Temp_dist[T_C, :]
                temp_pixel_id12 = Set(temp_pixel_id[Temp_dist1 <= reguired_objects_radius])
                temp_pixel_id1 = temp_pixel_id1 | temp_pixel_id12
            temp_pixel_id = Set(temp_pixel_id) - temp_pixel_id1
            temp_pixel_id = [ int(x) for x in temp_pixel_id ]

            temp_pixel_list1 = temp_pixel_list[:,temp_pixel_id]
            if len(temp_pixel_list1):
                Temp_centers, temp_ids = ss_take_sample(temp_pixel_list1, 'random', num_reguired_objects[nco] - 4)
            else:
                Temp_centers, temp_ids = ss_take_sample(temp_pixel_list, 'random', num_reguired_objects[nco] - 4)
            end_idc = int_idc + Temp_centers.shape[1]
            #print(int_idc, end_idc, Temp_centers)
            
            centers[:,int_idc:end_idc] = Temp_centers
            radii[0, int_idc:end_idc] = np.tile(reguired_objects_radius, (1, num_reguired_objects[nco] - 4))
            int_idc = end_idc
            #print(centers)

    centers = np.round(centers)
    radii = np.round(radii)
    
    return(centers, radii)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos):
    
    num_connected_object = len(connected_object_id)
    num_object = prev_pos.shape[1]
    object_id = np.array(np.arange(num_object))
    connected_idx = np.array([], dtype=int)
    non_connected_idx = np.array([], dtype=int)
    for nco in range(num_connected_object):
        Temp_center = np.asanyarray(stats[connected_object_id[nco]].centroid)
        Temp_center = np.reshape(Temp_center, (len(Temp_center), 1))
        Temp_dist = ss_euclidean_dist(Temp_center, prev_pos, 1)
        min_id = np.argmin(Temp_dist)
        Temp_center1 = prev_pos[:, min_id]
        Temp_pos = prev_pos[:,object_id!=min_id]
        Temp_dist1 = ss_euclidean_dist(Temp_center1, Temp_pos, 1)
        Temp_dist1 = Temp_dist1[Temp_dist1 <= 3.5*avg_radius]
        if len(Temp_dist1):
            connected_idx = np.concatenate((connected_idx, np.array([connected_object_id[nco]])))
        else:
            non_connected_idx = np.concatenate((non_connected_idx, np.array([connected_object_id[nco]])))
            
    return(connected_idx, non_connected_idx)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_1_connected_object(stats, num_notdetected_objects, centers, radii):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
            
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    num_division = np.array([2])    
    
    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_2_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
    if length_connected_object_id1 == 1:
        num_division =np.array([3])
        connected_object_id = connected_object_id1
    else:
        num_division = np.array([2, 2])
        connected_object_id = connected_object_id1
    connected_object_id.astype(int)
    
    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_3_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):

    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
        
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
    if length_connected_object_id1 == 1:
        num_division = np.array([4])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 2:
        num_division = np.array([3, 2])
        connected_object_id = connected_object_id1
    else:
        num_division = np.array([2, 2, 2])
        connected_object_id = connected_object_id1
    
    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_4_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
        
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
    if length_connected_object_id1 == 1:
        num_division = np.array([5])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 2:
        num_division = np.array([3, 3])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 3:
        num_division = np.array([3, 2, 2])
        connected_object_id = connected_object_id1
    else:
        num_division = np.array([2, 2, 2, 2])
        connected_object_id = connected_object_id1

    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_5_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
            
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
    if length_connected_object_id1 == 1:
        num_division = np.array([6])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 2:
        num_division = np.array([4, 3])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 3:
        num_division = np.array([4, 2, 2])#OR [3 3 3] depending on the major and minor axis check
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 4:
        num_division = np.array([3, 2, 2, 2])
        connected_object_id = connected_object_id1
                
    else:
        num_division = np.array([2, 2, 2, 2, 2])
        connected_object_id = connected_object_id1
        
    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_6_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
            
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
    if length_connected_object_id1 == 1:
        num_division = np.array([7])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 2:
        num_division = np.array([4, 4])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 3:
        num_division = np.array([4, 3, 2])#OR [3 3 3] depending on the major and minor axis check
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 4:
        num_division = np.array([4, 2, 2, 2])
        connected_object_id = connected_object_id1
    elif length_connected_object_id1 == 5:
        num_division = np.array([3, 2, 2, 2, 2])
        connected_object_id = connected_object_id1
                
    else:
        num_division = np.array([2, 2, 2, 2, 2, 2])
        connected_object_id = connected_object_id1
                
    return(centers, radii, connected_object_id, num_division)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_all_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos):
    
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    sort_id = np.argsort(Temp_area)
    connected_object_id = sort_id[-num_notdetected_objects:]
    non_connected_object_id = sort_id[0:-num_notdetected_objects]
            
    connected_object_id1, non_connected_object_id1 = ss_check_area_previous_frame(stats, connected_object_id, avg_radius, prev_pos)
    if len(non_connected_object_id1):
        non_connected_object_id = np.concatenate((non_connected_object_id, non_connected_object_id1))
    centers = centers[:,non_connected_object_id]
    radii = radii[non_connected_object_id]
    length_connected_object_id1 = len(connected_object_id1)
#     print('ss_all_connected_object %d %d \n' %(length_connected_object_id1, num_notdetected_objects))
#     if length_connected_object_id1 == 1:
#         num_division = np.array([8])
#         connected_object_id = connected_object_id1
#     elif length_connected_object_id1 == 2:
#         num_division = np.array([4, 4])
#         connected_object_id = connected_object_id1
#     elif length_connected_object_id1 == 3:
#         num_division = np.array([4, 3, 2])#OR [3 3 3] depending on the major and minor axis check
#         connected_object_id = connected_object_id1
#     elif length_connected_object_id1 == 4:
#         num_division = np.array([4, 2, 2, 2])
#         connected_object_id = connected_object_id1
#     elif length_connected_object_id1 == 5:
#         num_division = np.array([3, 2, 2, 2, 2])
#         connected_object_id = connected_object_id1
                
#     else:
    print('WE ARE FINDING SIX CONNECTED OBJECTS \n')
    num_division = np.zeros(length_connected_object_id1)
    count = 0
    Temp_count = length_connected_object_id1+num_notdetected_objects
    while(count < Temp_count):
        for temp_i in range(length_connected_object_id1):
            if(count < Temp_count):
                Temp_val = num_division[temp_i] + 1
                num_division[temp_i] = Temp_val
                count = count + 1
    num_division = num_division.astype(int)
            
    #num_division = np.array([2, 2, 2, 2, 2, 2])
    connected_object_id = connected_object_id1#[0:5]
        
    return(centers, radii, connected_object_id, num_division)
##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_object_detection(image, obj_method = ['binary','hist_max_channel', 50.0], background = np.array([]), num_objects = [], avg_radius = [], avg_area = [], prev_pos = np.array([])):
    
    check_flag = 0
    if(check_flag):
        print('Checking "ss_object_detection" \n')
    
    if not background.size:
        background = ss_backgound_estimation(image)
        
    if obj_method[0] == 'binary' :
        bin_image, level = ss_rgb_gray2binary(image, obj_method[1])
        area_thresh = obj_method[2]
        bin_image = ss_area_threshold(bin_image, area_thresh)
        
    else:
        bin_image = ss_backgound_substruction(image, background)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        bin_image = cv2.dilate(bin_image, kernel)

        area_thresh = obj_method[2]
        bin_image = ss_area_threshold(bin_image, area_thresh)
    
    stats, num_components = ss_connected_components(bin_image)
    
    centers = ss_list_with_attribute2array(stats, 'centroid')
    diameters = (ss_list_with_attribute2array(stats, 'major_axis_length') + ss_list_with_attribute2array(stats, 'minor_axis_length'))/2
    radii = diameters/2
    
    if num_objects:
        num_detected_objects = len(radii)
        num_notdetected_objects = num_objects - num_detected_objects
        #print('Temp-2 %d %d\n' %(num_objects, num_notdetected_objects))
        if num_detected_objects > num_objects:
            Temp_area = ss_list_with_attribute2array(stats, 'area')
            sort_id = np.argsort(Temp_area)
            Temp_object_id = sort_id[-num_objects-1:-1]
            Temp_stats = [None]*num_objects
            for nob in range(num_objects):
                Temp_stats[nob] = stats[Temp_object_id[nob]]
            stats = Temp_stats;
            centers = ss_list_with_attribute2array(stats, 'centroid')
            diameters = (ss_list_with_attribute2array(stats, 'major_axis_length') + ss_list_with_attribute2array(stats, 'minor_axis_length'))/2
            radii = diameters/2
            num_detected_objects = len(radii)
            num_notdetected_objects = num_objects - num_detected_objects
            
        if num_notdetected_objects == 0:
            centers = ss_list_with_attribute2array(stats, 'centroid')
            diameters = (ss_list_with_attribute2array(stats, 'major_axis_length') + ss_list_with_attribute2array(stats, 'minor_axis_length'))/2
            radii = diameters/2
            radii = np.reshape(radii,(1, len(radii)))
        elif num_notdetected_objects == 1:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            centers, radii, connected_object_id, num_division = ss_1_connected_object(stats, num_notdetected_objects, centers, radii)    
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
            
        elif num_notdetected_objects == 2:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            centers, radii, connected_object_id, num_division = ss_2_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
            
        elif num_notdetected_objects == 3:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            centers, radii, connected_object_id, num_division = ss_3_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
            
            
        elif num_notdetected_objects == 4:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            centers, radii, connected_object_id, num_division = ss_4_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
                
        elif num_notdetected_objects == 5:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            centers, radii, connected_object_id, num_division = ss_5_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
            
        elif num_notdetected_objects == 6:
            #print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
             
            centers, radii, connected_object_id, num_division = ss_6_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
            
        else:
            print('NOT DETECTED OBJECT %d \n' %num_notdetected_objects)
            
            print('WORNING: NUMBER OF OBJECTS IS BEYOND 6\n')
            
            centers, radii, connected_object_id, num_division = ss_all_connected_object(stats, num_notdetected_objects, centers, radii, avg_radius, prev_pos)
            Temp_centers, Temp_radii = ss_joint_object_separation(stats,  connected_object_id, num_division, avg_radius)
            
            centers = np.concatenate((centers, Temp_centers), axis=1)
            radii = np.reshape(radii,(1, len(radii)))
            radii = np.concatenate((radii, Temp_radii), axis= 1)
    centers = np.round(centers)
    radii = np.round(radii)
    
    return(background, bin_image, centers, radii, stats)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 22-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_NN_tracking_circle(images, save_path = [], save_flag = 0, disp_flag = 0, progress_flag = 0):

    check_flag = 0
    progress_flag = 0
    if(check_flag):
        print('Checking "ss_NN_tracking_circle" \n')
        
    traj_length = 130
    if save_path:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    num_frames = len(images)
    RES_images = []
    background, bin_image, centers, radii, stats = ss_object_detection(images[0])

    Temp_color_code = ss_color_generation(len(radii)+10)
    Temp_track_id = np.array(np.arange(len(radii)+10))
    
    track_pos = [None]*num_frames
    track_radius = [None]*num_frames
    track_id = [None]*num_frames
    
    RES_images = [None]*num_frames
    track_pos[0] = centers
    track_radius[0] = radii
    track_id[0] = np.array(np.arange(len(radii)))
    color_code = Temp_color_code[:,track_id[0]]
    Temp_color_code[:,0:len(track_id[0])] = color_code
    RES_images[0] = ss_draw_circlesOnImage(images[0], track_pos[0][[1,0],:], track_radius[0], color_code)
    
    text_id = Temp_track_id[track_id[0]]
    Temp_track_id[0:len(track_id[0])] = text_id
    text_id = ss_arrayelements2string(text_id)
    RES_images[0] = ss_draw_textOnImage(RES_images[0], text_id, track_pos[0][[1,0],:], [], color_code)
    
    num_objects = len(radii)
    avg_radius = radii[0]
    Temp_area = ss_list_with_attribute2array(stats, 'area')
    avg_area = np.mean(Temp_area)
    wolf_sheep_radius = 30*np.ones(2, np.int)
    wolf_sheep_color_code = np.array([[0, 255], [255, 0], [0, 0]], np.int)
    wolf_sheep_circumference_width = (-1)*np.ones(2, np.int)
    for nf in range(1,num_frames):
        if progress_flag:
            print('Processing frame # %d \n' % nf)
        background, bin_image, centers, radii, stats = ss_object_detection(images[nf], 'binary', background, num_objects, avg_radius, avg_area, track_pos[nf-1])
        if disp_flag:
            plt.figure(1)
            plt.imshow(bin_image, cmap='gray')
            plt.show()
        #print(track_pos[nf-1].shape)
        #print(centers.shape)
        match_ids = ss_point_match(track_pos[nf-1], centers)

        track_pos[nf] = centers
        track_radius[nf] = radii[0,:]
        track_id[nf] = match_ids
        color_code = Temp_color_code[:,match_ids]
        Temp_color_code[:,0:len(match_ids)] = color_code
        RES_images[nf] = ss_draw_circlesOnImage(images[nf], track_pos[nf][[1,0],:], track_radius[nf], color_code)
        
        text_id = Temp_track_id[match_ids]
        Temp_track_id[0:len(match_ids)] = text_id
        text_id = ss_arrayelements2string(text_id)
        RES_images[nf] = ss_draw_textOnImage(RES_images[nf], text_id, track_pos[nf][[1,0],:], [], color_code)

#         wolf_sheep_idx, wolf_sheep_traj = ss_match_trajectory(track_pos[:nf], track_id[:nf], traj_length)
#         if len(wolf_sheep_idx) > 1:
#             wolf_sheep_pos = track_pos[nf][:,wolf_sheep_idx]
# #             wolf_sheep_radius = (np.max(track_radius[nf][wolf_sheep_idx]) + 5)*np.ones(wolf_sheep_pos.shape[1], np.int)
# #             wolf_sheep_color_code = color_code[:,wolf_sheep_idx]
# #             wolf_sheep_circumference_width = (-1)*np.ones(wolf_sheep_pos.shape[1], np.int)
#             RES_images[nf] = ss_draw_circlesOnImage(images[nf], wolf_sheep_pos[[1,0],:], wolf_sheep_radius, wolf_sheep_color_code, wolf_sheep_circumference_width)
        if disp_flag:
            plt.figure(1)
            plt.imshow(RES_images[nf])
            plt.show()
            
        if save_flag:
            ss_images2video(1, RES_images, save_path)
            ss_images_write(RES_images, save_path)
            
    return(track_pos, track_id, track_radius, RES_images)

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

def ss_find_points_by_ids(obj_pos, obj_match_id, required_points_ids, max_num_points = 100):
    
    Temp_track_id = np.array(np.arange(max_num_points))
    num_frames = len(obj_match_id)
    num_points = len(required_points_ids)
    required_point_pos = [None]*num_frames
    for nf in range(num_frames):
        text_id = Temp_track_id[obj_match_id[nf]]
        Temp_track_id[0:len(obj_match_id[nf])] = text_id
        required_point_pos[nf] = np.zeros((2,num_points))
        #print('Frame %d \n' %nf)
        #print(len(obj_match_id[nf]), obj_match_id[nf])
        #print(text_id)
        #print(len(required_points_ids), required_points_ids)
        for npt in range(num_points):
            Temp_pos = obj_pos[nf][:, text_id==required_points_ids[npt]]
            #print(Temp_pos)
            required_point_pos[nf][:,npt] = np.reshape(Temp_pos,(len(Temp_pos)))
    return(required_point_pos)


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
def ss_find_moving_points(point_pos, traj_len = 25, dist_var_thresh = 0.00):
    num_frames = np.min((len(point_pos), traj_len))
    num_points = point_pos[0].shape[1]
    Temp_dist = np.zeros((num_frames-1, num_points))
    for nf in range(1, num_frames):
        Temp_dist[nf-1,:] = np.sqrt(np.sum(((point_pos[nf] - point_pos[nf-1])*(point_pos[nf] - point_pos[nf-1])),axis=0))
    #print(Temp_dist)
    Temp_dist_var = np.sum(Temp_dist, axis=0) 
    moving_points_id = np.array(range(num_points))
    moving_points_id = moving_points_id[Temp_dist_var > dist_var_thresh]
    
    num_frames = len(point_pos)
    moving_points = [None]*num_frames
    for nf in range(num_frames):
        moving_points[nf] = point_pos[nf][:,moving_points_id]
        
    moving_points_id = list(moving_points_id)
    
    return(moving_points_id, moving_points)

##########################################################################
def ss_find_moving_objects(point_pos, frame_number, dist_var_thresh = 0.0):
    
    if(frame_number < 2):
        print('ERROR: Please check the frame number %d \n' %frame_number)
        moving_points_id = []
    else:
        num_points = point_pos[frame_number].shape[1]
        Temp_dist = np.sqrt(np.sum(((point_pos[frame_number] - point_pos[frame_number-1])*(point_pos[frame_number] - point_pos[frame_number-1])),axis=0))
        moving_points_id = np.array(range(num_points))
        moving_points_id = moving_points_id[Temp_dist > dist_var_thresh]
        moving_points_id = list(moving_points_id)
        
    return(moving_points_id)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 20-04-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_initialization_blob_detection():
    
    ############PARAMETERS FOR BLOB DETECTORS################
    params = cv2.SimpleBlobDetector_Params()
 
    # Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.minInertiaRatio = 0.5
    detector = cv2.SimpleBlobDetector_create(params)
    
    return(params, detector)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 06-04-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_blob_detection(image, detector):
    
    blobs = detector.detect(image)
    num_blobs = len(blobs)
    key_points = np.zeros((2, num_blobs))
    radii = np.zeros((num_blobs))
    for snb in range(num_blobs):
        key_points[:,snb] = np.round(blobs[snb].pt)
        radii[snb] = np.round(blobs[snb].size)
    key_points = key_points[[1, 0], :]
    return(key_points, radii)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 06-04-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_blob_tracking(images, save_path = [], save_flag = 0, disp_flag = 0, progress_flag = 0):
    
    if save_path:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

##########PARAMETERS INITIALIZATION FOR BLOB DETECTORS################
    params, detector = ss_initialization_blob_detection()
    #################################################
#     radii = 5
    num_frames = len(images)
    centers, radii = ss_blob_detection(images[0], detector)
    num_objects = len(radii)
    
    Temp_color_code = ss_color_generation(num_objects+10)
    Temp_track_id = np.array(np.arange(num_objects+10))
    
    track_pos = [None]*num_frames
    track_radius = [None]*num_frames
    track_id = [None]*num_frames
    RES_images = [None]*num_frames
    
    track_pos[0] = centers
#     track_radius[0] = radii*np.ones(centers.shape[1])
    track_radius[0] = radii
    track_id[0] = np.array(np.arange(centers.shape[1]))
    color_code = Temp_color_code[:,track_id[0]]
    Temp_color_code[:,0:len(track_id[0])] = color_code
    RES_images[0] = ss_draw_circlesOnImage(images[0], track_pos[0][[1,0],:], track_radius[0], color_code)
    
    text_id = Temp_track_id[track_id[0]]
    Temp_track_id[0:len(track_id[0])] = text_id
    text_id = ss_arrayelements2string(text_id)
    RES_images[0] = ss_draw_textOnImage(RES_images[0], text_id, track_pos[0][[1,0],:], [], color_code)
    
    
    for snf in range(1, num_frames):
#         print(snf)
        centers, radii = ss_blob_detection(images[snf], detector)
        if len(radii) != num_objects:
            background, bin_image, centers, radii, stats = ss_object_detection(images[snf], ['binary','hist_max_channel', 50], np.array([]), num_objects, radii[0], [], track_pos[snf-1])
            radii = radii[0,:]
        match_ids = ss_point_match(track_pos[snf-1], centers)

        track_pos[snf] = centers
#         track_radius[snf] = radii*np.ones(centers.shape[1])
        track_radius[snf] = radii
        track_id[snf] = match_ids
        color_code = Temp_color_code[:,match_ids]
        Temp_color_code[:,0:len(match_ids)] = color_code
        RES_images[snf] = ss_draw_circlesOnImage(images[snf], track_pos[snf][[1,0],:], track_radius[snf], color_code)
        
        text_id = Temp_track_id[match_ids]
        Temp_track_id[0:len(match_ids)] = text_id
        text_id = ss_arrayelements2string(text_id)
        RES_images[snf] = ss_draw_textOnImage(RES_images[snf], text_id, track_pos[snf][[1,0],:], [], color_code)
    
        if disp_flag:
            plt.figure(1)
            plt.imshow(RES_images[snf])
            plt.show()
            
    if save_flag:
        ss_images2video(1, RES_images, save_path)
        ss_images_write(RES_images, save_path)
            
    return(track_pos, track_id, track_radius, RES_images) 

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 06-04-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_blob_tracking_kalman(images, save_path = [], save_flag = 0, disp_flag = 0, progress_flag = 0):
    
    if save_path:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
   
    ########PARAMETERS FOR BLOB DETECTORS################
    params, detector = ss_initialization_blob_detection()
    #################################################
#     radii = 5
    num_frames = len(images)
    centers, radii = ss_blob_detection(images[0], detector)
    num_objects = len(radii)
    
    ############FOR KALMAN INITIALIZATION##############
    s_X = [None]*num_objects
    s_state_error = [None]*num_objects
    s_phi = [None]*num_objects
    s_R = [None]*num_objects
    s_Z = [None]*num_objects
    s_H = [None]*num_objects
    s_Q = [None]*num_objects
    s_P = [None]*num_objects
    for sno in range(num_objects):        
        s_Z[sno] = np.reshape(centers[:,sno],(centers.shape[0], 1))
        s_X[sno], s_state_error[sno], s_phi[sno], s_Q[sno], s_H[sno], s_R[sno], s_P[sno] = ss_kalman_initialization(s_Z[sno])
        
    Temp_color_code = ss_color_generation(num_objects+10)
    Temp_track_id = np.array(np.arange(num_objects+10))
    
    track_pos = [None]*num_frames
    track_radius = [None]*num_frames
    track_id = [None]*num_frames
    RES_images = [None]*num_frames

    
    track_pos[0] = centers
    track_radius[0] = radii
    track_id[0] = np.array(np.arange(centers.shape[1]))
    color_code = Temp_color_code[:,track_id[0]]
    Temp_color_code[:,0:len(track_id[0])] = color_code
    RES_images[0] = ss_draw_circlesOnImage(images[0], track_pos[0][[1,0],:], track_radius[0], color_code)
    
    text_id = Temp_track_id[track_id[0]]
    Temp_track_id[0:len(track_id[0])] = text_id
    text_id = ss_arrayelements2string(text_id)
    RES_images[0] = ss_draw_textOnImage(RES_images[0], text_id, track_pos[0][[1,0],:], [], color_code)
    
    
    for snf in range(1, num_frames):
#         print(snf)
        centers, radii = ss_blob_detection(images[snf], detector)

        if len(radii) != num_objects:
            background, bin_image, centers, radii, stats = ss_object_detection(images[snf], ['binary','hist_max_channel', 50], np.array([]), num_objects, radii[0], [], track_pos[snf-1])
            radii = radii[0,:]
        Temp_centers = np.zeros((2, num_objects))
        Temp_id = list(range(centers.shape[1]))
        Temp_Temp_id = range(centers.shape[1])
        for sno in range(num_objects):
            s_X_pred, s_P_pred = ss_kalman_prediction(s_phi[sno], s_X[sno], s_state_error[sno], s_P[sno], s_Q[sno])
            P1 = np.reshape(s_X_pred[0:2], (centers.shape[0], ))
            sobject_detect = 0
            if len(Temp_Temp_id):
                P2 = centers[:,Temp_Temp_id]
                cost_matrix = ss_euclidean_dist(P1, P2, 1)
                mincost_id = np.argmin(cost_matrix)
                Temp_id.remove(Temp_Temp_id[mincost_id])
                Temp_Temp_id = [int(x) for x in Temp_id]
                s_Z[sno] = np.reshape(P2[:,mincost_id],(centers.shape[0], 1))
                
                if cost_matrix[0, mincost_id] < 30:
                    sobject_detect = 1
            if sobject_detect:
#                 s_X_pred, s_P_pred = ss_kalman_prediction(s_phi[sno], s_X[sno], s_state_error[sno], s_P[sno], s_Q[sno])
    #             print('s_X_pred', s_X_pred)
                s_X[sno], s_P[sno] = ss_kalman_update(s_X_pred, s_Z[sno], s_H[sno], s_R[sno], s_P_pred)
            else:
#                 s_X[sno], s_P[sno] = ss_kalman_prediction(s_phi[sno], s_X[sno], s_state_error[sno], s_P[sno], s_Q[sno])
                s_X[sno] = s_X_pred
                s_P[sno] = s_P_pred
                
            s_X[sno] = np.round(s_X[sno])
            Temp_centers[:,sno] = np.reshape(s_X[sno][0:2], (centers.shape[0], ))
        
        centers  = Temp_centers
        track_pos[snf] = centers
        track_radius[snf] = track_radius[0]
        match_ids = track_id[0]
        track_id[snf] = match_ids
        color_code = Temp_color_code[:,match_ids]
        Temp_color_code[:,0:len(match_ids)] = color_code
        RES_images[snf] = ss_draw_circlesOnImage(images[snf], track_pos[snf][[1,0],:], track_radius[snf], color_code)
        
        text_id = Temp_track_id[match_ids]
        Temp_track_id[0:len(match_ids)] = text_id
        text_id = ss_arrayelements2string(text_id)
        RES_images[snf] = ss_draw_textOnImage(RES_images[snf], text_id, track_pos[snf][[1,0],:], [], color_code)
    
        if disp_flag:
            image_disp_return_flag = ss_images_show([RES_images[snf]], 'frames#-'+str(snf))
            
    if save_flag:
        ss_images2video(1, RES_images, save_path)
        ss_images_write(RES_images, save_path)
            
    return(track_pos, track_id, track_radius, RES_images) 

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 21-05-18
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%----------------------------------------------------------------------------
#% EXAMPLE: 
#%NOTE: Currently, GOTURN has some problem and its not working!
##############################################################################
def ss_opencv_object_tracker(images, object_int_bbox, tracket_name='BOOSTING', tracker_bbox_draw=0):# tracket_name = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    
    if tracket_name == 'BOOSTING':
        object_tracker = cv2.TrackerBoosting_create()
    if tracket_name == 'MIL':
        object_tracker = cv2.TrackerMIL_create()
    if tracket_name == 'KCF':
        object_tracker = cv2.TrackerKCF_create()
    if tracket_name == 'TLD':
        object_tracker = cv2.TrackerTLD_create()
    if tracket_name == 'MEDIANFLOW':
        object_tracker = cv2.TrackerMedianFlow_create()
    if tracket_name == 'GOTURN':
        object_tracker = cv2.TrackerGOTURN_create() 
        
    # initialise the tracker    
    object_tracker.init(images[0], object_int_bbox)
    
    num_frames = len(images)
    tracked_info = [None]*num_frames
    tracked_info[0] = np.array([[int(object_int_bbox[0]), int(object_int_bbox[0] + object_int_bbox[2])],[int(object_int_bbox[1]), int(object_int_bbox[1] + object_int_bbox[3])]])                             
    
    # draw tracker bbox    
    if tracker_bbox_draw:  
        snf = 0
        images[snf] = cv2.rectangle(images[snf], (tracked_info[snf][0,0], tracked_info[snf][1,0]), (tracked_info[snf][0,1], tracked_info[snf][1,1]), (255,0,0), 2, 1)
        
    for snf in xrange(1, num_frames):
        img = images[snf]
        # main tracker api
        track_success_flag, tracked_bbox = object_tracker.update(img)
        # save the tracked info
        if track_success_flag:
            tracked_info[snf] = np.array([[int(tracked_bbox[0]), int(tracked_bbox[0] + tracked_bbox[2])],[int(tracked_bbox[1]), int(tracked_bbox[1] + tracked_bbox[3])]])                             
        else:
            print('tracker : {} failed to track at frame: {}' .format(tracket_name, snf))
            tracked_info[snf] = tracked_info[snf-1].copy()# take previous track info as current track info
        # draw tracker bbox    
        if tracker_bbox_draw:    
            images[snf] = cv2.rectangle(images[snf], (tracked_info[snf][0,0], tracked_info[snf][1,0]), (tracked_info[snf][0,1], tracked_info[snf][1,1]), (255,0,0), 2, 1)
        
    return tracked_info, images


