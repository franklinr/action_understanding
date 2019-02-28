
import os
import pickle
import time
import scipy.io as sio
import numpy as np
from xml.etree import ElementTree as ET

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
def ss_folder_create(save_path):
    if(len(save_path)):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return(save_path)
        
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
def ss_file_write(data_item, save_path='MATFILE_SAME/', file_name='mat_file'):
    
    save_path = ss_folder_create(save_path)
    file_id = open(save_path+file_name, "wb", -1)
    pickle.dump(data_item, file_id)
    file_id.close()
    return(1)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 180717
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_matfile_write(data_item, save_path='MATFILE_SAME/', file_name='mat_file'):
    
    save_path = ss_folder_create(save_path)
    sio.savemat(save_path+file_name+'.mat', data_item)
    return(1)


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
def ss_file_read(file_name):
    
    if os.path.isfile(file_name):
        file_id = open(file_name, 'rb')
        data_item = pickle.load(file_id)
        file_id.close()
    else:
        print('ERROR: No such file "%s" does not exits!!' %(file_name))
        data_item = []
    return(data_item)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 03-07-17
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_execution_time(func, *args):
    
    tic = time.time()
    func_value = func(*args)#Calculate function value
    toc = time.time()
    
    return (toc - tic), func_value

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 22-0618
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%NOTE: this function is copy and paste from http://effbot.org/zone/element-lib.htm#prettyprint
##########################################################################
def ss_xml_indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " \t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            ss_xml_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 22-0618
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_xml_tree_write(xml_tree, save_path='Temp_xml/', xml_file_name='xml', encoding_method='utf-8'):
    
    
    save_path = ss_folder_create(save_path)
    
    xml_tree.write('{}{}.xml'.format(save_path, xml_file_name, xml_declaration=True, encoding=encoding_method, method="xml"))
    
