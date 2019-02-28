
import numpy as np
import math
import cmath
#from cmath import rect, phase
#from math import radians, degrees

def ss_mean_angle(angles):
    return math.degrees(cmath.phase(sum(cmath.rect(1, math.radians(d)) for d in angles)/len(angles)))

########################################################################
#% FUNCTION: Check the zeros entries in a vector or set of vectors 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------------------
#% INPUT: X: X- A vector or set of vectors
#% OUTPUT: val: val - Zeros entries fillup by eps of X
#%--------------------------------------------------------------------------------------
#% EXAMPLE:
########################################################################
def ss_denominator_check(X):
    
    val = X
    if isinstance(val,(list, tuple, np.ndarray)):
         val[np.abs(val) == 0.0] = np.finfo(float).eps
    else:
        if val == 0:
            val = np.finfo(float).eps   
        
    return(val)

##########################################################################
#% FUNCTION: Calculate the euclidean distances between two set of vectors( X(dxm) and Y(dxn))
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, Y: Two set of vectors in matrix form X(dxm) and Y(dxn); where d is the dimension of
#%        each vector and m and n are the # vectors in set X and Y respectively.
#%        sqrt_flag (0 or 1): Flag for square root of the distance (default 1).
#% OUTPUT: dist_X_Y(mxn); where dist(i, j) is the euclidean distance between i-th and j-th
#%         vector of set X and Y respectively
#% NOTE: It also work for distance between two vectors x and y
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m, n = 100,2000, 1500
#%          X = np.random.random((d, m)
#%          Y = np.random.random((d, n))
#%          dist_X_Y = ss_euclidean_dist(X, Y, 1)
######################################################################
def ss_euclidean_dist(X, Y, sqrt_flag = 0):
    
    if np.ndim(X) == 1:
        X = np.reshape(X, (len(X), 1))
    if np.ndim(Y) == 1:
        Y = np.reshape(Y, (len(Y), 1))
    X_square = np.sum(X*X, axis=0)
    Y_square = np.sum(Y*Y, axis=0)
    X_Y = 2*np.dot(X.transpose(),Y)
    dist_X_Y = (np.reshape(X_square.transpose(),(X_square.shape[0], 1)) + Y_square)- X_Y
    dist_X_Y[dist_X_Y < 0.0] = 0
    if(( np.ndim(X) == 1) and (np.ndim(Y) == 1)):
        dist_X_Y = dist_X_Y[0,0]
    if sqrt_flag:
        dist_X_Y = np.sqrt(dist_X_Y)
        
    return(dist_X_Y)
    
############################################################################
#% FUNCTION: Linear scaling of set of vectors
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, scale_range: A set of vectors in matrix form X(dxm); where d is the dimension of
#%        each vector and m is the # vectors in set X. scale_range- scaling
#%        interval; scale_range(1)- min val and scale_range(2) - max_val
#% OUTPUT: data_scale: data_scale- is the scaled data of X within the range
#%         defined in scale_range
#% NOTE: It also work for single vector x
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100,2000
#%          scale_range = [0 1]
#%          X = np.random.random((d, m))
#%          data_scale = ss_linear_scale_data(X, scale_range)
############################################################################
def ss_linear_scale_data(X, scale_range, axis_flag = 1):

    if X.ndim <= 2:
        if ((X.ndim == 2) & (axis_flag==1)):
            min_val = np.reshape(np.min(X, axis= 0),(1, X.shape[1]))
            max_val = np.reshape(np.max(X, axis= 0),(1, X.shape[1]))
        else:
            min_val = np.min(X)
            max_val = np.max(X)
    else:
        if(axis_flag == 0):
            min_val = np.min(X)
            max_val = np.max(X)
        else:
            print('ERROR:SCALING FUNCTION IS NOT SUPORT FOR ARAY DIMS >3\n')
            return 0
    data_scale = scale_range[0] + (scale_range[1] - scale_range[0])*((X - min_val)/ss_denominator_check(max_val - min_val))
    
    return(data_scale)

###########################################################################    
#% FUNCTION: Data linear stretching
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, org_scale_range, tar_scale_range: A set of vectors in matrix form X(dxm); where d is the dimension of
#%        each vector and m is the # vectors in set X. org_scale_range- orininal scaling
#%        interval of data X (default org_scale_range(1),
#%        org_scale_range(2)- minimum and maximum value of X;
#%        tar_scale_range- required data scale. 
#% OUTPUT: stretch_data: stretch_data- is the scaled data of X within the range
#%         defined in tar_scale_range
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100, 2000
#%          org_scale_range = [0, 1]
#%          tar_scale_range = [0.5, 1]
#%          X = np.random.random((d, m))
#%          stretch_data = ss_linear_stretch(X, org_scale_range, tar_scale_range)
###########################################################################    
def ss_linear_stretch(X, org_scale_range, tar_scale_range):

    denom = org_scale_range[1] - org_scale_range[0]
    Temp1 = tar_scale_range[1] - tar_scale_range[0]
    Temp2 = org_scale_range[1]*tar_scale_range[0] - org_scale_range[0]*tar_scale_range[1]
    
    stretch_data = (Temp1*X + Temp2)/ss_denominator_check(denom)
    
    return(stretch_data)

##########################################################################
#% FUNCTION: Normalize a set of vectors with lp norm
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, l_norm: A set of vectors in matrix form X(dxm); where d is the dimension of
#%        each vector and m is the # vectors in set X. l_norm which norm are used for normalization 
#% OUTPUT: norml_vector: norml_vector- is the normalize vectors of
#%         X.
#% NOTE: It also work for single vector x 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100, 2000
#%          l_norm = 'l2'
#%          X = np.random.random((d, m))
#%          norml_vector = ss_lp_normalize_feature(X, l_norm)
##########################################################################
def ss_lp_normalize_feature(X, l_norm):       
    
    lp = int(l_norm[1:])
    Temp1 = np.reshape(np.sum(X**lp, axis=0)**(1.0/lp),(1, X.shape[1]))
    normat_vector = X/ss_denominator_check(Temp1)
    
    return(normat_vector) 

##########################################################################    
#% FUNCTION: Normalize a set of vectors with mean zeros and standard deviation one
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, mu, sigma: A set of vectors in matrix form X(dxm); where d is the dimension of
#%        each vector and m is the # vectors in set X. mu and sigma are the mean and 
#%        standard of the data respectively. By default these are calculated from data. 
#% OUTPUT: norml_vector, mu, sigma: norml_vector- is the normalize vectors of
#%         X. mu and sigma are the mean and standard of the data
#%         respectively.
#% NOTE: It also work for single vector x 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100, 2000
#%          X = np.random.random((d, m)) 
#%          [norml_vector, mu, sigma] = ss_mean_std_normalize(X)
#%
##########################################################################
def ss_mean_std_normalize(X, mu, sigma):
    
    if not mu:
        mu = np.reshape(np.mean(X, axis=1),(X.shape[0], 1))
    if not sigma:
        sigma = np.reshape(np.std(X, axis=1),(X.shape[0], 1))
        
    norml_vector = (X - mu)/ss_denominator_check(sigma)
    
    return(norml_vector, mu, sigma)

###########################################################################
#% FUNCTION: 2-3-4D random mask with zero mean and variance one
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%-----------------------------------------------------------------------------------------
#% INPUT: mask_size, number_mask:
#% OUTPUT: masks
#%-----------------------------------------------------------------------------------------
#% EXAMPLE: masks = ss_ND_random_mask(mask_size, number_mask)
##########################################################################

def ss_ND_random_mask(mask_size, number_mask):

    if not number_mask:
        number_mask = 1
    mask_size.append(number_mask)
    masks = np.random.random(mask_size)
    for nm in range(number_mask):
        Temp = masks[...,nm]
        Temp = Temp - np.mean(Temp)
        Temp = Temp/ss_denominator_check(np.std(Temp))
        masks[...,nm] = Temp
        
    return(masks)
###########################################################################
#% FUNCTION: Calculate softmax of a set of vectors
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X: X- input data of size (dx1 or dxN or MxNxd or MxNxdxP) where d is
#% the dimention of the data and do softmax accordingly
#% OUTPUT: softmax_val: softmax_val- softmax value of X
#% NOTE:
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100, 2000
#%          X = np.random.random((d, m)) 
#%          softmax_val= ss_softmax(X)
#%
##########################################################################
def ss_softmax(X):
    
    num_dims = X.ndim
    if num_dims == 2:
        max_val = np.reshape(np.max(X, axis=0),(1, X.shape[1]))
        numerator_val = np.exp(X - max_val)
        denominator_val = np.reshape(np.sum(numerator_val, axis=0),(1, numerator_val.shape[1]))
        softmax_val = numerator_val/ss_denominator_check(denominator_val)
    elif num_dims == 3:
        max_val = np.reshape(np.max(X, axis=2),(X.shape[0], X.shape[1], 1))
        numerator_val = np.exp(X - max_val)
        denominator_val = np.reshape(np.sum(numerator_val, axis=2),(numerator_val.shape[0], numerator_val.shape[1], 1))
        softmax_val = numerator_val/ss_denominator_check(denominator_val)
    else:
        print('ERROR:PLEASE CHECK THE DATA DIMENSION (<=3)\n')
        return 0
        
    return(softmax_val)
    
###########################################################################
#% FUNCTION: Take random samples (with replacement or without replacement) from
#%  a set of data
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: X, sample_type, num_samples, replacement_flag: X- input data of size 
#%    (1xN or dxN or dxmxN or dxmxnxNP) where N is the number of data and do 
#%      sampling accordingly; sample_type- sample types ('random' or 'uniform') 
#%    num_samples- requied number of samples (default 1);
#%  replacement_flag- sample technique (with or without replacement) (default 0);
#% OUTPUT: samples, ids: samples- required number of samples from X; ids- samples 
#%  ids of the oiginal data.
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m = 100, 2000
#%          X = np.random.random((d, m)) 
#%          (samples, ids) = ss_take_sample(data, 'random', 10, 0)
#%
##########################################################################
def ss_take_sample(X, sample_type = 'random', num_samples = 1, replacement_flag = 0):
    
    num_points = int(X.shape[-1])
    if num_points < num_samples:        
        print('WORNING: NUMBER REQUIRED SAMPLE (%d) IS > DATA SAMPLES (%d) \n' %(num_samples, num_points) )
        num_samples = num_points
    if sample_type == 'random':
        temp_dims = list(range(X.ndim))
        for i in range(X.ndim):
            temp_dims[i] = X.shape[i]
        temp_dims[-1] = num_samples

        if replacement_flag:
            samples = np.zeros(temp_dims)
            ids = list(range(num_samples))
            for ns in range(num_samples):
                rand_id = np.random.permutation(num_points)
                ids[ns] = rand_id[0];
                samples[..., ns] = X[..., ids[ns]]
        else:
            rand_id = np.random.permutation(num_points)
            ids = rand_id[0:num_samples]
            samples = X[..., ids]
    if sample_type == 'uniform':
        sample_interval = float(num_points)/num_samples
        ids = np.array(np.arange(0, num_points, sample_interval),int)
        samples = X[..., ids]
        
    return(samples, ids)
    
###########################################################################
#% FUNCTION: Generate 2D random points (non-overlap) from an 2D grid 
#% WRITER: SOUMITRA SAMANTA            DATE: 02-11-16/30-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: num_points, row, column, replacement_flag, dx_row, dx_column, dx, dy: 
#%        num_points- number of points to be generated; row, column- size of the 2D grid (default, 320x420);
#%        replacement_flag- sample technique (with or without replacement) (default 0);
#%        dx_row, dx_column, dx, dy- different flag to handle border points
#% OUTPUT: pos: pos- required number of 2D points
#%--------------------------------------------------------------------------------------
#% EXAMPLE: num_points = 100
#%          pos = ss_random_2D_points(num_points)
#%
##########################################################################
def ss_random_2D_points(num_points, row = 320, column = 420, replacement_flag = 0, dx_row = 5, dx_column = 5, dx = 11, dy = 11):

    row_low = dx_row
    row_up = row - dx_row - 1
    column_low = dx_column
    column_up = column - dx_column - 1
    x = np.arange(row_low, row_up, dx)
    y = np.arange(column_low, column_up, dy)
    xv, yv = np.meshgrid(x, y)
    data = np.concatenate((np.reshape(xv,(1, xv.size)), np.reshape(yv,(1, yv.size)))) 
    (pos, ids) = ss_take_sample(data, 'random', num_points, replacement_flag)
    
    return(pos)
    
###########################################################################
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
def ss_random_trajectory_gen_rtheta(int_pos, row, column, dx_row, dx_column, radius, theta, tajectory_length):

    pos = np.zeros((2, tajectory_length))
    row_low = dx_row
    row_up = row - dx_row - 1
    column_low = dx_column
    column_up = column - dx_column - 1
        
    pos[:,0] = np.reshape(int_pos,(int_pos.shape[0],))
    for tl in range(1, tajectory_length):
        temp_radius, idr = ss_take_sample(radius, 'random', 1, 0)
        temp_theta, idt = ss_take_sample(theta, 'random', 1, 1)
        x = np.round(temp_radius*np.cos(temp_theta))
        y = np.round(temp_radius*np.sin(temp_theta))
        t_x = pos[0,tl-1] + x
        if t_x < row_low:
            t_x = row_low
        if t_x > row_up:
            t_x = row_up
        pos[0,tl] = t_x
        t_y = pos[1,tl-1] + y
        if t_y < column_low:
            t_y = column_low
        if t_y > column_up:
            t_y = column_up
        pos[1,tl] = t_y
        
    return(pos)




##########################################################################
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
def ss_equation_circle(points):

    if ((points.shape[0] == 2) & (points.shape[1] == 3)):  
        x1, y1 = points[0,0], points[1,0]
        x2, y2 = points[0,1], points[1,1]
        x3, y3 = points[0,2], points[1,2]
        delta = (x1-x2)*(y2-y3) - (x2-x3)*(y1-y2);
        delta_1 = y1*(x2**2-x3**2 + y2**2-y3**2) + y2*(x3**2-x1**2 + y3**2-y1**2) + y3*(x1**2-x2**2 + y1**2-y2**2)
        delta_2 = x1*(x2**2-x3**2 + y2**2-y3**2) + x2*(x3**2-x1**2 + y3**2-y1**2) + x3*(x1**2-x2**2 + y1**2-y2**2)

        Temp_center = np.array((delta_1, -delta_2))/ss_denominator_check(delta)
        c = -x1**2-y1**2 - (x1*delta_1-y1*delta_2)/ss_denominator_check(delta)
        radius = np.sqrt(np.sum((Temp_center**2)/4) -c)
        center = -Temp_center/2.0
    else:
        print('ERROR:PLEASE ENTER THE THREE POINTS\n')
        return(0, 0)
        
    return(center, radius)

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
# def ss_list2array(data, field_val):
    
#     if hasattr(data[0], field_val):
#         num_list_element = len(data)
#         out_data = []
#         for nle in range(num_list_element):
#             out_data.append(getattr(data[nle], field_val))            
#         out_data = np.asanyarray(out_data)
#         out_data = np.transpose(out_data)
        
#         return(out_data)
#     else:
#         print('ERROR: NO ATTRIBUTE NAME "%s" FOUND \n' %field_val)
#         return(0)
def ss_list2array(data): #260318
    
    out_data = np.asanyarray(data)
    out_data = np.rollaxis(out_data, 0, np.ndim(out_data))
    
    return out_data

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 04-04-18
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_list_with_attribute2array(data, field_val):
    
    if hasattr(data[0], field_val):
        num_list_element = len(data)
        out_data = []
        for nle in range(num_list_element):
            out_data.append(getattr(data[nle], field_val))            
        out_data = np.asanyarray(out_data)
        out_data = np.transpose(out_data)
        
        return(out_data)
    else:
        raise ValueError('ERROR: NO ATTRIBUTE NAME "%s" FOUND \n' %field_val)
        return(0)

    
##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 14-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_array2list(data, axis_data = 0):
    
    if np.ndim(data) > axis_data:
        num_list_element = data.shape[axis_data]
        out_data = [None]*num_list_element
        if axis_data == 0:
            for nle in range(num_list_element):
                out_data[nle] = data[nle,...]
        elif axis_data == 1:
            for nle in range(num_list_element):
                out_data[nle] = data[:,nle,...]
        elif axis_data == 2:
            for nle in range(num_list_element):
                out_data[nle] = data[:,:,nle,...]
        elif axis_data == 3:
            for nle in range(num_list_element):
                out_data[nle] = data[:,:,:,nle,...]
        elif axis_data == 4:
            for nle in range(num_list_element):
                out_data[nle] = data[:,:,:,:,nle,...]
        else:
            raise ValueError('WARNING: CREATING LIST BASED ON LAST AXIS "%d" (PLEASE MODIFY THE CODE FOR HIGHER AXIS (<5)) \n' %np.ndim(data))
            for nle in range(num_list_element):
                out_data[nle] = data[...,nle]
        
        return(out_data)
    else:
        raise ValueError('ERROR: AXIS %d IS OUTOF DATA AXIS %d \n' %(axis_data, np.ndim(data)))
        return(0)
    
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
def ss_arrayelements2string(data, str_len = 2):
    
    num_dimension = data.ndim
    str_data = np.chararray((data.shape),itemsize = str_len)
    if num_dimension == 1:
        for i in range(data.shape[0]):
            str_data[i] = str(data[i])
    elif num_dimension == 2:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                str_data[i, j] = str(data[i, j])
    else:
        raise ValueError('ERROR:DATA DIMENSION IS NOT SUPORTED (#dim<3) \n')
        return([])
    return(str_data)             
        
##############################################################################
#% FUNCTION: Matching two sets of points
#% WRITER: SOUMITRA SAMANTA            DATE: 30-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: A, B, match_frag: A, B- two input sets of points of size 
#%    (dxM and dxN) where M, N are the number of data points at A and B respectively with dimension d;
#%      match_frag- matching type (default nearest neighbour (nn)
#% OUTPUT: match_id: match_id- match id of B with A (Ex. match_id[0] = 3 means B[:,0] point is matched 
#%         with the A[:,3] point)
#%--------------------------------------------------------------------------------------
#% EXAMPLE: d, m, n = 100, 2000, 3000
#%          A = np.random.random((d, m)) 
#%          B = np.random.random((d, n)) 
#%          match_id = ss_point_match(A, B)
#%
##########################################################################    
def ss_point_match(A, B, match_frag = 'nn'):

    num_points = A.shape[1]
    Temp_num_points = B.shape[1]
    
    match_id = np.zeros(Temp_num_points)
    if match_frag == 'nn':
        if num_points >= Temp_num_points:
            Temp_id = list(range(num_points))
            Temp_Temp_id = range(num_points)
            for npoint in range(Temp_num_points):
                P1 = B[:,npoint]
                P2 = A[:,Temp_Temp_id]
                cost_matrix = ss_euclidean_dist(P1, P2, 1)
                mincost_id = np.argmin(cost_matrix)
                match_id[npoint] = Temp_Temp_id[mincost_id]
                Temp_id.remove(Temp_Temp_id[mincost_id])
                Temp_Temp_id = [int(x) for x in Temp_id]
        else:
            Temp_id = list(range(Temp_num_points))
            Temp_Temp_id = range(Temp_num_points)
            for npoint in range(num_points):
                    P1 = A[:,npoint]
                    P2 = B[:,Temp_Temp_id]
                    cost_matrix = ss_euclidean_dist(P1, P2, 1)
                    mincost_id = np.argmin(cost_matrix)
                    match_id[Temp_Temp_id[mincost_id]] = npoint
                    Temp_id.remove(Temp_Temp_id[mincost_id])
                    Temp_Temp_id = [int(x) for x in Temp_id]
            Temp_match_id = num_points
            for npoint in Temp_Temp_id:
                match_id[npoint] = Temp_match_id
                Temp_match_id+=1
    else:
        print('ONLY WORK FOR MATCHING METHODS LIKE ("nn") \n')
        return(0)
    return(match_id.astype(int))

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
def ss_anglefrom2points(P1, P2, degree_flag = 0):
    
    P1 = np.reshape(P1, (2,))
    P2 = np.reshape(P2, (2,))
    T = P2-P1
    point_angle = np.arctan2(T[1], T[0])
    if degree_flag:
        point_angle = point_angle*(180.0/np.pi) 
        
    return(point_angle)
    #angle = atan2(norm(cross(a,b)), dot(a,b))
    
##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 29-06-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_cosine_anglefrom2vectors(Va, Vb, degree_flag = 1):
    
    Va = np.reshape(Va, (Va.shape[0], 1))
    Vb = np.reshape(Vb, (Vb.shape[0], 1))
    Va = ss_lp_normalize_feature(Va, 'l2')
    Vb = ss_lp_normalize_feature(Vb, 'l2')
    
    dot_Va_Vb = np.sum(Va*Vb)
    ###------MAKE SURE -1<=cos_theta<=1 --------#####
    if dot_Va_Vb > 1.0:
        dot_Va_Vb = 1.0
    if dot_Va_Vb < -1.0:
        dot_Va_Vb = -1.0       
    cos_angle = math.acos(dot_Va_Vb)
    if degree_flag:
        cos_angle = cos_angle*(180.0/np.pi)       
        
    return(cos_angle)
    
##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 31-05-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################    
def ss_theta2vector(theta):
    
    if np.isscalar(theta):
        pos = np.zeros((2, 1))
        Temp_theta = theta*(np.pi/180.0)
        pos[0, 0] = np.cos(Temp_theta)
        pos[1, 0] = np.sin(Temp_theta)
        
    else:
        num_theta = theta.shape[1]
        pos = np.zeros((2, num_theta))
        for snt in range(num_theta):
            Temp_theta = theta[0, snt]*(np.pi/180.0)
            pos[0, snt] = np.cos(Temp_theta)
            pos[1, snt] = np.sin(Temp_theta)
    return(pos)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 16-06-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################  
def ss_expand_2d_to_3d_matrix(X, row, col, heigh, verbose_flag = 0):
    
    expand_X = np.zeros((row, col, heigh))
    
    if X.ndim == 2:
        if((X.shape[0] <= row) & (X.shape[1] <= col)):
            num_objects = X.shape[0]
            cell_size_row = np.int(np.floor(row/X.shape[0]))
            cell_size_col = np.int(np.floor(col/X.shape[1]))
            Temp_id_row = np.arange(0, row, cell_size_row)
            Temp_id_col = np.arange(0, col, cell_size_col)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    expand_X[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,:] = X[i, j]*np.ones((cell_size_row, cell_size_col, heigh))
        else:
            if verbose_flag:
                print('TARGET MATRIX (%d, %d) SMALLER THAN SOURCE MATRIX (%d, %d)' %(row, col, X.shape[0], X.shape[1]))
    expand_X = np.squeeze(expand_X)
    
    return(expand_X)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 19-06-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################  
#def ss_take_velocity_direction(num_points, int_theta = 0.0, end_theta = 360.0, theta_div = 1.0, sample_type = 'random'):
#    
#    theta = np.arange(int_theta, end_theta, theta_div) # velocity direction range
#    velo_theta, idt = ss_take_sample(theta, sample_type, num_points, 1) # random velocity direction
#    int_pos_velo = np.zeros((2,num_points))
#    Temp_pos_theta = velo_theta*(np.pi/180.0)
#    int_pos_velo[0,:] = np.cos(Temp_pos_theta)
#    int_pos_velo[1,:] = np.sin(Temp_pos_theta)
#   
#    return(int_pos_velo)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 200717
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################  
def ss_vector_scaling(pos_vector, scale_mag):

#         pos_vector_direction = ss_lp_normalize_feature(pos_vector, 'l2')
        pos_vector_direction = normalize(pos_vector, axis=0)
        scaled_pos_vector = scale_mag*pos_vector_direction

        return(scaled_pos_vector)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 12-04-18
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################  
def ss_matrix_compare(X1, X2, axis_flag=0):
    if(np.ndim(X1) == np.ndim(X2)):
        Temp_ids = np.equal(X1, X2)
        ids = np.array(xrange(Temp_ids.shape[axis_flag])).astype(int)
        if axis_flag==0:
            Temp_sum = np.sum(Temp_ids, axis=1)
            equa_ids = ids[Temp_sum==Temp_ids.shape[1]]
            diffr_ids = ids[Temp_sum!=Temp_ids.shape[1]]
        else:
            Temp_sum = np.sum(Temp_ids, axis=0)
            equa_ids = ids[Temp_sum==Temp_ids.shape[0]]
            diffr_ids = ids[Temp_sum!=Temp_ids.shape[0]]
        
    else:
        raise ValueError('Matrices must be same dimension X1: {}; X2: {}' .format(X1.shape, X2.shape))
        
    return equa_ids, diffr_ids


