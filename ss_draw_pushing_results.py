###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 06122018
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
import time
import matplotlib as plt
from IPython.display import clear_output
from sklearn.preprocessing import normalize
##############################################################################################

import os
import sys
import getpass
import uuid
#For my all purpose code path
mac_address = hex(uuid.getnode())#gives mac address of your computer(ex for my mac: 0x784f4391fc66, HP server: 0xcc47a69614bL)
user_name = getpass.getuser()#gives your current system username
 
if((mac_address=='0x784f4391fc66') & (user_name=='soumitra')):#for my mac
    ss_lib_path = '/Users/soumitra/Documents/SS_CODE/SS_PYTHON/SS_PYLIBS/'
    
elif((mac_address=='0xcc47a69614bL') & (user_name=='samanta')):#for liverpool HP-server
    ss_lib_path = '/media/big/samanta/SS_CODE/SS_PYTHON/SS_PYLIBS/'
    
elif os.path.exists('SS_PYLIBS/'):#for current folder
    ss_lib_path = 'SS_PYLIBS/'
    
else:
    raise ValueError('Add the soumitra all purpose ("SS_PYLIBS") code path with following:\n mac_address: {};\n user_name: {} ' .format(mac_address, user_name));
    
sys.path.insert(0, ss_lib_path)
from ss_computation import *
from ss_image_video import *
from ss_drawing import *
from ss_input_output import *
from ss_object_tracking import *
from ss_human_action_data_gen_helper import *
from ss_action_heuristics import *
##############################################################################################

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2
##############################################################################################
pushing_data_gen_date = '02122018'
pushing_test_data_date = '06122018'
action_name = 'pushing'
num_objects = 9
max_speed = 5.0
pushing_contact_distance = [0, 10, 20]
pushing_delay = [0, 10, 20]
pushing_heuristics = 'Avg velocity before and after touch'

input_result_path = '' .join(['classification_results/data/', pushing_data_gen_date, '/', action_name, '_nobj_', str(num_objects),'/images/'])
input_result_file_name = '' .join([input_result_path, 'all_pushing_result_based_on_distance_nobj_',  str(num_objects), '_mspeed_', str(int(max_speed)), '_heurist_', pushing_heuristics.replace(' ','_'), '_date_', pushing_test_data_date, '.npz'])

data = np.load(input_result_file_name)
result = data['final_result']

#plot based on delay and contact distance
Temp_result = result.mean(axis=0).mean(axis=0)
print('pushing results based on delay and distance: {}' .format(Temp_result))

#plot based on delay  
plt.title('Pushing result')
for i in range(Temp_result.shape[0]):
    Temp_legend = 'contact distance: {}'.format(pushing_contact_distance[i]) # put the heuristics name here
    plt.plot(Temp_result[i,:], '-o', label=Temp_legend)

    plt.xticks(np.arange(Temp_result.shape[1]), pushing_delay)
    plt.xlabel('Pushing delay')

    yticks_min = 10*(np.int(max(0, np.min(Temp_result[i,:])-20))/10)
    yticks_max = 101
    yticks_interval = 10
    plt.yticks(np.arange(yticks_min, yticks_max, yticks_interval))
    plt.ylabel('Accuracy (%)')

    plt.grid(linestyle='--')
    plt.legend(loc='upper right')
##############################################################################################

#save plot results
Temp_file_name = '' .join([input_result_path, 'pushing_result_based_on_delay_nobj_',  str(num_objects), '_mspeed_', str(int(max_speed)), '_heurist_', pushing_heuristics.replace(' ','_'), '_date_', pushing_test_data_date])
plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

plt.show(block=False)
plt.close()
##############################################################################################

#plot based on contact distance  
plt.title('Pushing result')
for i in range(Temp_result.shape[1]):
    Temp_legend = 'delay: {}'.format(pushing_delay[i]) # put the heuristics name here
    plt.plot(Temp_result[:,i], '-o', label=Temp_legend)

    plt.xticks(np.arange(Temp_result.shape[0]), pushing_contact_distance)
    plt.xlabel('Pushing contact distance')

    yticks_min = 10*(np.int(max(0, np.min(Temp_result[:,i])-20))/10)
    yticks_max = 101
    yticks_interval = 10
    plt.yticks(np.arange(yticks_min, yticks_max, yticks_interval))
    plt.ylabel('Accuracy (%)')

    plt.grid(linestyle='--')
    plt.legend(loc='upper right')
##############################################################################################

#save plot results
Temp_file_name = '' .join([input_result_path, 'pushing_result_based_on_distance_nobj_',  str(num_objects), '_mspeed_', str(int(max_speed)), '_heurist_', pushing_heuristics.replace(' ','_'), '_date_', pushing_test_data_date])
plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

plt.show(block=False)
plt.close()
##############################################################################################
