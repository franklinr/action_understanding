
import numpy as np


##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 25-03-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_kalman_initialization(s_Z_0, del_t = 1):
    
    s_phi = np.array(([[1,0,del_t,0],[0,1,0,del_t],[0,0,1,0],[0,0,0,1]]),'d') #Transition matrix
    s_Q = 0.001*np.eye((4)) # Transition covarience
    s_H = np.array(([[1,0,0,0],[0,1,0,0]]),'d') #Observation matrix
    s_R = np.array(([[0.05, 0.01],[0.01, 0.05]]),'d') # Observation covarience
    s_P = 100.0*np.eye(4) # Initial covarience
    s_X = np.array(([s_Z_0[0], s_Z_0[1], 0, 0]),'d')
    s_X = np.reshape(s_X, (s_X.size, 1))
    s_state_error = np.array(([0.0, 0.0, 0.0, 0.0]),'d') #Error in stateprint(s_phi)
    s_state_error = np.reshape(s_state_error, (s_state_error.size, 1))
    
    return(s_X, s_state_error, s_phi, s_Q, s_H, s_R, s_P)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 25-03-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_kalman_prediction(s_phi, s_X, s_state_error, s_P, s_Q):
    
    s_X_pred = np.dot(s_phi, s_X) + s_state_error
    s_P_pred = np.dot(s_phi, np.dot(s_P, s_phi.T)) + s_Q
    
    return(s_X_pred, s_P_pred)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 25-03-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_kalman_update(s_X_pred, s_Z, s_H, s_R, s_P_pred):
    
    Temp_var = np.dot(s_H, np.dot(s_P_pred, s_H.T)) + s_R
    K = np.dot(s_P_pred, np.dot(s_H.T, np.linalg.inv(Temp_var)))
    s_P_update = np.dot((np.eye(4) - np.dot(K, s_H)), s_P_pred)
    s_X_update = s_X_pred + np.dot(K, (s_Z - np.dot(s_H, s_X_pred)))
    
    return(s_X_update, s_P_update)


