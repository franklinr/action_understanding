
###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 200717/120817/1407817/180917/031017/140218/220218/260418/050918
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


class ss_pushing(object):
    
    def __init__(self, args):
        
        self.todaysdate = time.strftime("%d%m%Y")#give the date in ddmmyyyy format

        ##parameters of the chasing action
        self.pushing_subtelty_angle = args.pushing_subtelty_angle# range of pushing subtelties
        if not isinstance(self.pushing_subtelty_angle, np.ndarray):
            self.pushing_subtelty_angle = [self.pushing_subtelty_angle]
        self.num_pushing = args.num_pushing#different number of pushing in a video
        if not isinstance(self.num_pushing, np.ndarray):
            self.num_pushing = [self.num_pushing]
        self.pushing_contact_distance = args.pushing_contact_distance#distance between pusher and pushe dusing pushing
        if not isinstance(self.pushing_contact_distance, np.ndarray):
            self.pushing_contact_distance = [self.pushing_contact_distance]
        self.pushing_delay = args.pushing_delay#pushing delay in number of frames
        if not isinstance(self.pushing_delay, np.ndarray):
            self.pushing_delay = [self.pushing_delay]
        self.push_interval = args.push_interval#gap between two pushing
        self.Temp_max_speed = [args.max_speed]# speed of each velocity
        self.Temp_max_steer_force = [args.max_steer_force]#steering force corresponding to the object speed
        self.num_objects = args.push_num_objects  #number of circle at each video
        self.object_radius = args.object_radius # object radius (As used by Gao)
        self.object_circumference_width = args.object_circumference_width #circular object circumference width (As used by Gao)
        #object_color = args.object_color[1:-1].replace(' ','').split(',')
        #self.object_color = np.reshape(np.array([int(object_color[0]), int(object_color[1]), int(object_color[2])]),(3, 1))#object color (As used by Gao)
        self.object_color = np.reshape(np.array([args.object_color, args.object_color, args.object_color]),(3, 1))#object color (As used by Gao)
        self.row = args.push_row#video frame size (As used by Gao)
        self.column = args.push_column#video frame size (As used by Gao)
        self.image_background = args.image_background#video frame background color
        #self.tajectory_length = args.tajectory_length #number of frames in a video
        self.num_video_files = args.num_video_files#how many video files want to generate
        self.object_init_reuse_flag = args.object_init_reuse_flag#initial object position taken from saved position
        self.object_init_reuse_folder = args.object_init_reuse_folder##initial object position taken path
        self.position_saving_flag = args.position_saving_flag# object position saving flag
        self.video_saving_flag = args.video_saving_flag# Video save flag
        self.video_saving_flag_label = args.video_saving_flag_label # Video save flag with object label
        self.ougput_video_file_extention = args.ougput_video_file_extention# output video file format
        self.data_save_path = args.data_save_path# data save path
        if not len(self.data_save_path):
            self.data_save_path = 'data/{}/{}_nobj_{}/'.format(self.todaysdate, args.action_name, self.num_objects)
        self.decimal_file_number = args.decimal_file_number# ouput file number in decimal
        self.verbose_flag = args.verbose_flag# vervose flag
        if not len(self.object_init_reuse_folder):
            self.object_init_reuse_folder = '{}pos/'.format(self.data_save_path)
            
    
    def data_generation(self):
        print('##############################################################################################')
        print('generating pushing data')
        ##############################################################################################
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        for stms in range(len(self.Temp_max_speed)):
            for sps in self.pushing_subtelty_angle:
                for snpu in self.num_pushing:
                    for spcd in self.pushing_contact_distance:
                        for spd in self.pushing_delay:

                            ########DIFFERENT PARAMETERS FOR DATA GENERATIONS###############
                            tajectory_length = snpu*(200+self.push_interval) + 2*self.push_interval #number of frames in a video
                            dx_row, dx_column = 5*self.object_radius + 1, 5*self.object_radius + 1 #handling image boundaries
                            dx, dy = 2*dx_row + 1, 2*dx_column + 1 # grid cells for 2d pont sample
                            int_theta = 0.0
                            end_theta = 360.0
                            theta_div = 1.0
                            replacement_flag = 0 # random sample replacement flag
                            max_speed = self.Temp_max_speed[stms]*np.ones((1, self.num_objects)) # constant velocity of each objects(best 6.0)
                            max_steer_force = max_speed/2.5 #3.5*np.ones((1, self.num_objects)) # maximum steering force(best max_speed/2.5)
                            # disjoint_percentage = 1 # between(0-overlap to 1-non-overlap); object overlap factor
                            disjoint_threshold = 8.0*self.object_radius*np.ones((1, self.num_objects))#sp.reshape(int(disjoint_percentage*(3*(2*self.object_radius+1))+1)*np.ones((1, self.num_objects)),(1, self.num_objects))
                            velo_theta_thresh = 30.0*np.ones((1, self.num_objects)) #velocity moving direction range from perious frame.
                            #############FOR DISPLAY AND DATA SAVING###############
                            #data_save_path = 'Result/{0}/object_motion_pushing_NOB_{1}/pushing_subtlety_{2}_nop_{3}_maxspeed_{4}/' .format(self.todaysdate, self.num_objects, int(sps/2.0), snpu, int(max_speed[0,0]))
                            ########################################################

                            for snv in range(0, self.num_video_files):
                                print('----------------------------------------------------------------------------------------------')
                                print('generating {0}(/{1})-th video data for tms {2} ps {3} npu {4} pcd {5} and pd {6}' .format(snv, self.num_video_files, stms, sps/2.0, snpu, spcd, spd))
                                Temp_file_name_mask = 'ps_{}_np_{}_mspeed_{}_spcd_{}_spd_{}_{}' .format(int(sps/2.0), int(snpu), int(self.Temp_max_speed[stms]), spcd, spd, Temp_file_number%snv)
                                if self.verbose_flag:
                                    print('INITIALIZE THE OBJECTS POSITION AT FIRST FRAME.....')
                                if  self.object_init_reuse_flag:
                                    Temp_file_name = '{}pos_{}.p'.format(self.object_init_reuse_folder, Temp_file_name_mask)
                                    int_pos = ss_file_read(Temp_file_name)
                                if(not  self.object_init_reuse_flag or len(int_pos) == 0):
                                    int_pos = ss_random_2D_points(self.num_objects, self.row, self.column, replacement_flag, 2*dx_row, 2*dx_column, dx, dy) # random posions in first frame
                                else:
                                    int_pos = int_pos[:,:,0].copy()
                                if  self.object_init_reuse_flag:
                                    Temp_file_name = '{}pos_velo_{}.p'.format(self.object_init_reuse_folder, Temp_file_name_mask)
                                    int_pos_velo_direc = ss_file_read(Temp_file_name)
                                if(not  self.object_init_reuse_flag or len(int_pos_velo_direc) == 0):
                                    int_pos_velo_direc = ss_init_velocity_direction(self.num_objects)# random velocity direction
                                else:
                                    int_pos_velo_direc = int_pos_velo_direc[:,:, 0].copy()
                                    int_pos_velo_direc = ss_lp_normalize_feature(int_pos_velo_direc, 'l2')
                            #     print(int_pos_velo_direc)
                                int_pos_velo = max_speed*int_pos_velo_direc #initial velocity
                            #     print(int_pos_velo)

                                ########TAKE WOLF MAX DISTANCE FROM SHEEP########
                                Temp_dist = ss_euclidean_dist(np.reshape(int_pos[:,0], (2, 1)).copy(), int_pos.copy())
                                Temp_X = int_pos[:,1].copy()
                                int_pos[:,1] = int_pos[:,Temp_dist.argmax()].copy()
                                int_pos[:,Temp_dist.argmax()] = Temp_X.copy()
                                #################################################

                                # call actual data generation api
                                if self.verbose_flag:
                                    print('WORKING ACTUAL DATA GENERATING FUNCTION.....')
                                pos, pos_velo, pos_push_frame_info, pos_push_id = ss_pushing_random_trajectory(int_pos, int_pos_velo, sps, snpu, spcd, spd, self.push_interval, max_speed, max_steer_force, self.row, self.column, dx_row, dx_column, self.object_radius, tajectory_length, velo_theta_thresh, disjoint_threshold)
                                
                                #generate a video
                                images = ss_frame_generate(self.row, self.column, self.image_background, pos, self.object_radius*np.ones((self.num_objects), np.int32), self.object_color, self.object_circumference_width*np.ones((self.num_objects), np.int))
                                #################################################

                                # save the generated data
                                if self.verbose_flag:
                                    print('SAVING THE RESULT at: {}' .format(self.data_save_path))
                                    
                                # for position and others information
                                if self.position_saving_flag:
                                    Temp_folder_name = '{}pos/'.format(self.data_save_path)
                                    Temp_file_name = 'pos_{}.p'.format(Temp_file_name_mask)
                                    ss_file_write(pos, Temp_folder_name, Temp_file_name)# for position info
                                    Temp_file_name = 'pos_velo_{}.p'.format(Temp_file_name_mask)
                                    ss_file_write(pos_velo, Temp_folder_name, Temp_file_name)# for velocity info
                                    Temp_file_name = 'pos_push_frame_info_{}.p'.format(Temp_file_name_mask)
                                    ss_file_write(pos_push_frame_info, Temp_folder_name, Temp_file_name)# for pushing frame info
                                    Temp_file_name = 'pos_push_id_{}.p'.format(Temp_file_name_mask)
                                    ss_file_write(pos_push_id, Temp_folder_name, Temp_file_name)# for pusher and pushe info
                                    
                                # for video file
                                if(self.video_saving_flag):
                                    Temp_folder_name = '{}video/without_label/'.format(self.data_save_path)
                                    Temp_file_name = 'video_file_{}'.format(Temp_file_name_mask)
                                    ss_images2video(0, images, Temp_folder_name, Temp_file_name, self.ougput_video_file_extention)# for video without label info
                                    
                                    #for object label
                                    if(self.video_saving_flag_label):
                                        text_id = ss_arrayelements2string(np.array(range(self.num_objects)))
                                        num_frames = len(images)
                                        for snt in xrange(num_frames):
                                            images[snt] = ss_draw_textOnImage(images[snt], text_id, np.reshape(pos[[1, 0],:,snt],(pos.shape[0], pos.shape[1])))
                                        # save object label video
                                        Temp_folder_name = '{}video/with_label/'.format(self.data_save_path)
                                        Temp_file_name = 'video_file_{}'.format(Temp_file_name_mask)
                                        ss_images2video(0, images, Temp_folder_name, Temp_file_name, self.ougput_video_file_extention)
        print('##############################################################################################')
                                            

    def data_classification_org(self, args):

        print('##############################################################################################')
        print('testing pushing based on original position data')
        ##############################################################################################
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        pos_data_path = '{}pos/'.format(self.data_save_path)
        video_data_path = '{}video/without_label/'.format(self.data_save_path)
        classification_data_save_path = args.classification_data_save_path
        if not len(classification_data_save_path):
            classification_data_save_path = 'classification_results/{}/'.format(self.data_save_path)
        
        action_start_frame_id = args.pushing_action_start_frame_id
        frame_diff_thresh = args.pushing_frame_diff_thresh
        interval_thgreh = args.pushing_avg_speed_frame_threshold
        pushing_heuristics = 'Avg velocity before and after touch'
        pushing_test_data_date = args.pushing_test_data_date
        
        video_label_accuracy = [None]*len(self.pushing_subtelty_angle)
        pushing_subtelty_count = 0
        ##############################################################################################
        for stms in range(len(self.Temp_max_speed)):
            for sps in self.pushing_subtelty_angle:#loops on pushing subtlety
                video_label_accuracy[pushing_subtelty_count] = [None]*len(self.num_pushing)
                num_pushing_count = 0
                for snpu in self.num_pushing:#loops on number of pushing with in a video
                    video_label_accuracy[pushing_subtelty_count][num_pushing_count] = [None]*len(self.pushing_contact_distance)
                    num_pushing_contact_distance_count = 0
                    for spcd in self.pushing_contact_distance:#loops on distance between puher and pushee
                        video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count] = [None]*len(self.pushing_delay)
                        num_pushing_delay_count = 0
                        for spd in self.pushing_delay:#loops on pushing delay
                            video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count] = np.zeros((4, self.num_video_files))
                            for snv in range(0, self.num_video_files):
                                print('----------------------------------------------------------------------------------------------')
                                print('classifying {0}(/{1})-th video data for tms {2} ps {3} npu {4} pcd {5} and pd {6}' .format(snv, self.num_video_files, stms, sps/2.0, snpu, spcd, spd))
                                Temp_file_name_mask = 'ps_{}_np_{}_mspeed_{}_spcd_{}_spd_{}_{}' .format(int(sps/2.0), int(snpu), int(self.Temp_max_speed[stms]), spcd, spd, Temp_file_number%snv)
                                ##############################################################################################
                                
                                # read original position info
                                Temp_file_name = '{}pos_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos = ss_file_read(Temp_file_name)
                                Temp_file_name = '{}pos_push_frame_info_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos_push_frame_info = ss_file_read(Temp_file_name)
                                Temp_file_name = '{}pos_push_id_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos_push_id = ss_file_read(Temp_file_name)
                                
                                org_pushing_info = np.concatenate((pos_push_id, pos_push_frame_info[2:,:]), axis=0).astype(int)
                                if((spcd==0.0) and (spd==0.0)):
                                    org_class_ids = 1
                                else:
                                    org_class_ids = 0
                                    
                                # read video file
                                Temp_file_name = '{}video_file_{}.{}'.format(video_data_path, Temp_file_name_mask, self.ougput_video_file_extention)
                                images = ss_video2images(Temp_file_name)
                                #for object label
                                if(self.video_saving_flag_label):
                                    text_id = ss_arrayelements2string(np.array(range(self.num_objects)))
                                    num_frames = len(images)
                                    for snt in xrange(num_frames):
                                        images[snt] = ss_draw_textOnImage(images[snt], text_id, np.reshape(pos[[1, 0],:,snt],(pos.shape[0], pos.shape[1])))
                                ##############################################################################################
                                
                                # main heuristics (To do)
                                est_touch_info, est_pushing_class_ids, est_pushing_info, est_pushing_score = ss_pushing_detection(pos, self.num_objects, self.object_radius, frame_diff_thresh, interval_thgreh)
                                est_pushing_score = est_pushing_score[-1]
                                # check with original pusher
                                if org_class_ids:
                                    est_pushing_match_count, est_pushing_match_ids = ss_pushing_match(org_pushing_info, est_pushing_info, frame_diff_thresh=10)
                                    est_pushing_class_ids = np.sum(est_pushing_match_count==org_pushing_info.shape[1])

                                video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count][2, snv] = est_pushing_class_ids                        
                                ##############################################################################################
                                
                                avg_final_score = est_pushing_score[org_pushing_info[0,:],org_pushing_info[1,:],-1].mean()
                                if avg_final_score > 1.:
                                    avg_final_score = 1.
                                video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count][3, snv] = avg_final_score

                                # save the results
                                Temp_folder_name = '{}chache/'.format(classification_data_save_path)
                                Temp_folder_name = ss_folder_create(Temp_folder_name)
                                Temp_file_name = '{}est_push_info_{}'.format(Temp_folder_name, Temp_file_name_mask)
                                np.savez(Temp_file_name, est_pushing_class_ids=est_pushing_class_ids, est_pushing_info=est_pushing_info, est_pushing_score=est_pushing_score)
                                ##############################################################################################
                                
                                # for video saving
                                if(self.video_saving_flag):
                                    # add the cost matrix
                                    num_frames = est_pushing_score.shape[2]
                                    for i in range(num_frames):
                                        cost_img = ss_image_create_cost_matrix(est_pushing_score[:,:,i], cost_image_size=[300, 300], display_flag='area', title='Pushing Score Map')
                                        T_image = (255*np.ones((images[i].shape[0]-cost_img.shape[0], cost_img.shape[1], 3))).astype('uint8')
                                        T_image = np.concatenate((cost_img, T_image), axis=0)
                                        images[i] = np.concatenate((images[i], T_image), axis=1)
                                    ##############################################################################################

                                    # mark pushing objects
                                    if est_pushing_class_ids:
                                        test_color = ss_color_generation(est_pushing_info.shape[1])#np.reshape(np.array([255, 0, 0]),(3, 1))
                                        for i in xrange(est_pushing_info.shape[1]):
                                            text_id = np.array(['ESTP_{}'.format(i+1)])
                                            images[est_pushing_info[2,i]] = ss_draw_circlesOnImage(images[est_pushing_info[2,i]], np.reshape(pos[:,est_pushing_info[:2,i], est_pushing_info[2,i]],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            # put text on lower object
                                            text_pos = pos[:,est_pushing_info[0,i], est_pushing_info[2,i]] - pos[:,est_pushing_info[1,i], est_pushing_info[2,i]]
                                            if(text_pos[0]>0):
                                                text_pos = pos[:,est_pushing_info[0,i], est_pushing_info[2,i]].copy() 
                                            else:
                                                text_pos = pos[:,est_pushing_info[1,i], est_pushing_info[2,i]].copy() 
                                            text_pos[0] += 10
                                            text_pos[1] += self.object_radius+5
                                            text_pos = np.reshape(text_pos, (2,1))[[1,0],:]

                                            images[est_pushing_info[2,i]] = ss_draw_textOnImage(images[est_pushing_info[2,i]], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)

                                            # mark all pushing info at the end frame
                                            images[-1] = ss_draw_circlesOnImage(images[-1], np.reshape(pos[:,est_pushing_info[:2,i], -1],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            text_id = np.array(['ESTP_{}-->'.format(i+1), 'ESTP_{}<--'.format(i+1)])
                                            # put text on lower object
                                            text_pos = np.reshape(pos[:,est_pushing_info[:2,i], -1],(2, 2))
                                            text_pos[0,:] += 15
                                            text_pos[1,:] += self.object_radius+5
                                            text_pos = text_pos[[1,0],:]
                                            images[-1] = ss_draw_textOnImage(images[-1], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)
                                    ##############################################################################################

                                    if org_class_ids:
                                        test_color = ss_color_generation(org_pushing_info.shape[1])#np.reshape(np.array([255, 0, 0]),(3, 1))
                                        for i in xrange(org_pushing_info.shape[1]):
                                            text_id = np.array(['ORGP_{}'.format(i+1)])
                                            images[org_pushing_info[2,i]] = ss_draw_circlesOnImage(images[org_pushing_info[2,i]], np.reshape(pos[:,org_pushing_info[:2,i], org_pushing_info[2,i]],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            # put text on lower object
                                            text_pos = pos[:,org_pushing_info[0,i], org_pushing_info[2,i]] - pos[:,org_pushing_info[1,i], org_pushing_info[2,i]]
                                            if(text_pos[0]>0):
                                                text_pos = pos[:,org_pushing_info[0,i], org_pushing_info[2,i]].copy() 
                                            else:
                                                text_pos = pos[:,org_pushing_info[1,i], org_pushing_info[2,i]].copy() 
                                            text_pos[0] += 10
                                            text_pos[1] -= self.object_radius+30
                                            text_pos = np.reshape(text_pos, (2,1))[[1,0],:]

                                            images[org_pushing_info[2,i]] = ss_draw_textOnImage(images[org_pushing_info[2,i]], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)

                                            # mark all pushing info at the end frame
                                            images[-1] = ss_draw_circlesOnImage(images[-1], np.reshape(pos[:,org_pushing_info[:2,i], -1],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            text_id = np.array(['ORGP_{}-->'.format(i+1), 'ORGP_{}<--'.format(i+1)])
                                            # put text on lower object
                                            text_pos = np.reshape(pos[:,org_pushing_info[:2,i], -1],(2, 2))
                                            text_pos[0,:] -= 15
                                            text_pos[1,:] += self.object_radius+5
                                            text_pos = text_pos[[1, 0],:]

                                            images[-1] = ss_draw_textOnImage(images[-1], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)
                                    ##############################################################################################

                                    # save video file
                                    # save only 1 video per constraint
                                    if snv== 0:
                                        Temp_folder_name = '{}video/'.format(classification_data_save_path)
                                        Temp_file_name = 'video_file_{}'.format(Temp_file_name_mask)
                                        ss_images2video(0, images, Temp_folder_name, Temp_file_name, self.ougput_video_file_extention)# for video without label info
                                    # save last frame of each videos
                                    Temp_folder_name = '{}video/last_frame/'.format(classification_data_save_path)
                                    Temp_file_name = 'last_frame_video_file_{}'.format(Temp_file_name_mask)
                                    ss_images_write([images[-1]], save_path=Temp_folder_name, file_name=Temp_file_name)
                                ##############################################################################################
                            num_pushing_delay_count += 1
                            ##############################################################################################
                        num_pushing_contact_distance_count += 1
                        ##############################################################################################
                    num_pushing_count += 1
                    ##############################################################################################
                pushing_subtelty_count += 1
                ##############################################################################################
        
        #calculating pushing score accuracy
        pushing_subtelty_count = len(self.pushing_subtelty_angle)
        num_pushing_count = len(video_label_accuracy[0])
        num_pushing_contact_distance_count = len(video_label_accuracy[0][0])
        num_pushing_delay_count = len(video_label_accuracy[0][0][0])
        result = np.zeros((num_pushing_count, pushing_subtelty_count, num_pushing_contact_distance_count, num_pushing_delay_count))
        for i1 in range(num_pushing_count):
            total_count = 0.0
            correct_est_count = 0.0
            for j1 in range(pushing_subtelty_count):
                for k1 in range(num_pushing_contact_distance_count):
                    for l1 in range(num_pushing_delay_count):
                        Temp = video_label_accuracy[j1][i1][k1][l1]                   
                        total_count += 1
                        result[i1, j1, k1, l1] = Temp[3,:].mean()
        result *= 100.
        ##############################################################################################

        #plot based on delay and contact distance
        Temp_result = result.mean(axis=0).mean(axis=0)
        print('pushing results based on delay and distance: {}' .format(Temp_result))

        #plot based on delay  
        plt.title('Pushing result')
        for i in range(Temp_result.shape[0]):
            Temp_legend = 'contact distance: {}'.format(self.pushing_contact_distance[i]) # put the heuristics name here
            plt.plot(Temp_result[i,:], '-o', label=Temp_legend)

            plt.xticks(np.arange(Temp_result.shape[1]), self.pushing_delay)
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
        Temp_folder_name = '{}images/'.format(classification_data_save_path)
        Temp_folder_name = ss_folder_create(Temp_folder_name)
        Temp_file_name = '{0}pushing_result_based_on_delay_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
        plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

        plt.show(block=False)
        plt.close()
        ##############################################################################################

        #plot based on contact distance  
        plt.title('Pushing result')
        for i in range(Temp_result.shape[1]):
            Temp_legend = 'delay: {}'.format(self.pushing_delay[i]) # put the heuristics name here
            plt.plot(Temp_result[:,i], '-o', label=Temp_legend)

            plt.xticks(np.arange(Temp_result.shape[0]), self.pushing_contact_distance)
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
        Temp_folder_name = '{}images/'.format(classification_data_save_path)
        Temp_folder_name = ss_folder_create(Temp_folder_name)
        Temp_file_name = '{0}pushing_result_based_on_distance_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
        plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

        plt.show(block=False)
        plt.close()
        ##############################################################################################

        final_result = result
        Temp_file_name = '{0}all_pushing_result_based_on_distance_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        np.savez(Temp_file_name, final_result=final_result, video_label_accuracy=video_label_accuracy)
        ##############################################################################################  

        print('##############################################################################################')
    ##############################################################################################

    def data_classification_track(self, args):

        print('##############################################################################################')
        print('testing pushing based on tracking position data')
        ##############################################################################################
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        pos_data_path = '{}pos/'.format(self.data_save_path)
        video_data_path = '{}video/without_label/'.format(self.data_save_path)
        classification_data_save_path = args.classification_data_save_path
        if not len(classification_data_save_path):
            classification_data_save_path = 'classification_results/{}/'.format(self.data_save_path)
        
        action_start_frame_id = args.pushing_action_start_frame_id
        frame_diff_thresh = args.pushing_frame_diff_thresh
        interval_thgreh = args.pushing_avg_speed_frame_threshold
        pushing_heuristics = 'Avg velocity before and after touch'
        pushing_test_data_date = args.pushing_test_data_date
        
        video_label_accuracy = [None]*len(self.pushing_subtelty_angle)
        pushing_subtelty_count = 0
        ##############################################################################################
        for stms in range(len(self.Temp_max_speed)):
            for sps in self.pushing_subtelty_angle:#loops on pushing subtlety
                video_label_accuracy[pushing_subtelty_count] = [None]*len(self.num_pushing)
                num_pushing_count = 0
                for snpu in self.num_pushing:#loops on number of pushing with in a video
                    video_label_accuracy[pushing_subtelty_count][num_pushing_count] = [None]*len(self.pushing_contact_distance)
                    num_pushing_contact_distance_count = 0
                    for spcd in self.pushing_contact_distance:#loops on distance between puher and pushee
                        video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count] = [None]*len(self.pushing_delay)
                        num_pushing_delay_count = 0
                        for spd in self.pushing_delay:#loops on pushing delay
                            video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count] = np.zeros((4, self.num_video_files))
                            for snv in range(0, self.num_video_files):
                                print('----------------------------------------------------------------------------------------------')
                                print('classifying {0}(/{1})-th video data for tms {2} ps {3} npu {4} pcd {5} and pd {6}' .format(snv, self.num_video_files, stms, sps/2.0, snpu, spcd, spd))
                                Temp_file_name_mask = 'ps_{}_np_{}_mspeed_{}_spcd_{}_spd_{}_{}' .format(int(sps/2.0), int(snpu), int(self.Temp_max_speed[stms]), spcd, spd, Temp_file_number%snv)
                                ##############################################################################################
                                
                                # read original position info
                                Temp_file_name = '{}pos_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos = ss_file_read(Temp_file_name)
                                Temp_file_name = '{}pos_push_frame_info_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos_push_frame_info = ss_file_read(Temp_file_name)
                                Temp_file_name = '{}pos_push_id_{}.p'.format(pos_data_path, Temp_file_name_mask)
                                pos_push_id = ss_file_read(Temp_file_name)
                                
                                org_pushing_info = np.concatenate((pos_push_id, pos_push_frame_info[2:,:]), axis=0).astype(int)
                                if((spcd==0.0) and (spd==0.0)):
                                    org_class_ids = 1
                                else:
                                    org_class_ids = 0
                                    
                                # read video file
                                Temp_file_name = '{}video_file_{}.{}'.format(video_data_path, Temp_file_name_mask, self.ougput_video_file_extention)
                                images = ss_video2images(Temp_file_name)
                                
                                ##############################################################################################
                    
                                #call object tracking api
                                if(self.verbose_flag):
                                    print('Tracking progressing.....');
                                track_images = copy.deepcopy(images)
                                tracked_pos, obj_match_id, track_radius, track_images = ss_blob_tracking_kalman(track_images)

                                #Adjust the tracking position datastructure according to the original positions datastructure
                                num_frames = len(tracked_pos)
                                num_points = tracked_pos[0].shape[1]
                                Temp_tracked_pos = np.zeros((tracked_pos[0].shape[0], num_points, num_frames))
                                for ttsnp in range(num_frames):
                                    Temp_tracked_pos[:,:,ttsnp] = tracked_pos[ttsnp]

                                #match the tracking object identities with the original object identities
                                compare_frame_ids = 0
                                tttt = ss_point_match(Temp_tracked_pos[:,:,compare_frame_ids], pos[:,:,compare_frame_ids])
                                tracked_pos = Temp_tracked_pos[:,tttt,:]

                                #generate a tracked video
                                images = ss_frame_generate(self.row, self.column, self.image_background, tracked_pos, self.object_radius*np.ones((self.num_objects), np.int32), self.object_color, self.object_circumference_width*np.ones((self.num_objects), np.int))
                                #for object label
                                if(self.video_saving_flag_label):
                                    text_id = ss_arrayelements2string(np.array(range(self.num_objects)))
                                    num_frames = len(images)
                                    for snt in xrange(num_frames):
                                        images[snt] = ss_draw_textOnImage(images[snt], text_id, np.reshape(pos[[1, 0],:,snt],(pos.shape[0], pos.shape[1])))
                                ##############################################################################################
                                
                                
                                # main heuristics (To do)
                                est_touch_info, est_pushing_class_ids, est_pushing_info, est_pushing_score = ss_pushing_detection(tracked_pos, self.num_objects, self.object_radius, frame_diff_thresh, interval_thgreh)
                                est_pushing_score = est_pushing_score[-1]
                                # check with original pusher
                                if org_class_ids:
                                    est_pushing_match_count, est_pushing_match_ids = ss_pushing_match(org_pushing_info, est_pushing_info, frame_diff_thresh=10)
                                    est_pushing_class_ids = np.sum(est_pushing_match_count==org_pushing_info.shape[1])

                                video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count][2, snv] = est_pushing_class_ids                        
                                ##############################################################################################
                                #est_pushing_score /= 2.#only for tracking based
                                avg_final_score = est_pushing_score[org_pushing_info[0,:],org_pushing_info[1,:],-1].mean()
                                if avg_final_score > 1.:
                                    avg_final_score = 1.
                                video_label_accuracy[pushing_subtelty_count][num_pushing_count][num_pushing_contact_distance_count][num_pushing_delay_count][3, snv] = avg_final_score

                                # save the results
                                Temp_folder_name = '{}chache/'.format(classification_data_save_path)
                                Temp_folder_name = ss_folder_create(Temp_folder_name)
                                Temp_file_name = '{}tracked_est_push_info_{}'.format(Temp_folder_name, Temp_file_name_mask)
                                np.savez(Temp_file_name, est_pushing_class_ids=est_pushing_class_ids, est_pushing_info=est_pushing_info, est_pushing_score=est_pushing_score)
                                ##############################################################################################

                                # for video saving
                                if(self.video_saving_flag):
                                    # add the cost matrix
                                    num_frames = est_pushing_score.shape[2]
                                    for i in range(num_frames):
                                        cost_img = ss_image_create_cost_matrix(est_pushing_score[:,:,i], cost_image_size=[300, 300], display_flag='area', title='Pushing Score Map')
                                        T_image = (255*np.ones((images[i].shape[0]-cost_img.shape[0], cost_img.shape[1], 3))).astype('uint8')
                                        T_image = np.concatenate((cost_img, T_image), axis=0)
                                        images[i] = np.concatenate((images[i], T_image), axis=1)
                                    ##############################################################################################

                                    # mark pushing objects
                                    if est_pushing_class_ids:
                                        test_color = ss_color_generation(est_pushing_info.shape[1])#np.reshape(np.array([255, 0, 0]),(3, 1))
                                        for i in xrange(est_pushing_info.shape[1]):
                                            text_id = np.array(['ESTP_{}'.format(i+1)])
                                            images[est_pushing_info[2,i]] = ss_draw_circlesOnImage(images[est_pushing_info[2,i]], np.reshape(pos[:,est_pushing_info[:2,i], est_pushing_info[2,i]],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            # put text on lower object
                                            text_pos = pos[:,est_pushing_info[0,i], est_pushing_info[2,i]] - pos[:,est_pushing_info[1,i], est_pushing_info[2,i]]
                                            if(text_pos[0]>0):
                                                text_pos = pos[:,est_pushing_info[0,i], est_pushing_info[2,i]].copy() 
                                            else:
                                                text_pos = pos[:,est_pushing_info[1,i], est_pushing_info[2,i]].copy() 
                                            text_pos[0] += 10
                                            text_pos[1] += self.object_radius+5
                                            text_pos = np.reshape(text_pos, (2,1))[[1,0],:]

                                            images[est_pushing_info[2,i]] = ss_draw_textOnImage(images[est_pushing_info[2,i]], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)

                                            # mark all pushing info at the end frame
                                            images[-1] = ss_draw_circlesOnImage(images[-1], np.reshape(pos[:,est_pushing_info[:2,i], -1],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            text_id = np.array(['ESTP_{}-->'.format(i+1), 'ESTP_{}<--'.format(i+1)])
                                            # put text on lower object
                                            text_pos = np.reshape(pos[:,est_pushing_info[:2,i], -1],(2, 2))
                                            text_pos[0,:] += 15
                                            text_pos[1,:] += self.object_radius+5
                                            text_pos = text_pos[[1,0],:]
                                            images[-1] = ss_draw_textOnImage(images[-1], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)
                                    ##############################################################################################

                                    if org_class_ids:
                                        test_color = ss_color_generation(org_pushing_info.shape[1])#np.reshape(np.array([255, 0, 0]),(3, 1))
                                        for i in xrange(org_pushing_info.shape[1]):
                                            text_id = np.array(['ORGP_{}'.format(i+1)])
                                            images[org_pushing_info[2,i]] = ss_draw_circlesOnImage(images[org_pushing_info[2,i]], np.reshape(pos[:,org_pushing_info[:2,i], org_pushing_info[2,i]],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            # put text on lower object
                                            text_pos = pos[:,org_pushing_info[0,i], org_pushing_info[2,i]] - pos[:,org_pushing_info[1,i], org_pushing_info[2,i]]
                                            if(text_pos[0]>0):
                                                text_pos = pos[:,org_pushing_info[0,i], org_pushing_info[2,i]].copy() 
                                            else:
                                                text_pos = pos[:,org_pushing_info[1,i], org_pushing_info[2,i]].copy() 
                                            text_pos[0] += 10
                                            text_pos[1] -= self.object_radius+30
                                            text_pos = np.reshape(text_pos, (2,1))[[1,0],:]

                                            images[org_pushing_info[2,i]] = ss_draw_textOnImage(images[org_pushing_info[2,i]], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)

                                            # mark all pushing info at the end frame
                                            images[-1] = ss_draw_circlesOnImage(images[-1], np.reshape(pos[:,org_pushing_info[:2,i], -1],(2, 2))[[1, 0],:], (self.object_radius+5)*np.ones(2), circle_color=test_color[:,i:i+1], circumference_width=self.object_circumference_width)

                                            text_id = np.array(['ORGP_{}-->'.format(i+1), 'ORGP_{}<--'.format(i+1)])
                                            # put text on lower object
                                            text_pos = np.reshape(pos[:,org_pushing_info[:2,i], -1],(2, 2))
                                            text_pos[0,:] -= 15
                                            text_pos[1,:] += self.object_radius+5
                                            text_pos = text_pos[[1, 0],:]

                                            images[-1] = ss_draw_textOnImage(images[-1], text_id, text_pos, text_color=test_color[:,i:i+1], text_font_size=1, text_width=2)
                                    ##############################################################################################

                                    # save video file
                                    # save only 1 video per constraint
                                    if snv== 0:
                                        Temp_folder_name = '{}video/'.format(classification_data_save_path)
                                        Temp_file_name = 'tracked_video_file_{}'.format(Temp_file_name_mask)
                                        ss_images2video(0, images, Temp_folder_name, Temp_file_name, self.ougput_video_file_extention)# for video without label info
                                    # save last frame of each videos
                                    Temp_folder_name = '{}video/last_frame/'.format(classification_data_save_path)
                                    Temp_file_name = 'last_frame_tracked_video_file_{}'.format(Temp_file_name_mask)
                                    ss_images_write([images[-1]], save_path=Temp_folder_name, file_name=Temp_file_name)
                                ##############################################################################################
                            num_pushing_delay_count += 1
                            ##############################################################################################
                        num_pushing_contact_distance_count += 1
                        ##############################################################################################
                    num_pushing_count += 1
                    ##############################################################################################
                pushing_subtelty_count += 1
                ##############################################################################################
        
        #calculating pushing score accuracy
        pushing_subtelty_count = len(self.pushing_subtelty_angle)
        num_pushing_count = len(video_label_accuracy[0])
        num_pushing_contact_distance_count = len(video_label_accuracy[0][0])
        num_pushing_delay_count = len(video_label_accuracy[0][0][0])
        result = np.zeros((num_pushing_count, pushing_subtelty_count, num_pushing_contact_distance_count, num_pushing_delay_count))
        for i1 in range(num_pushing_count):
            total_count = 0.0
            correct_est_count = 0.0
            for j1 in range(pushing_subtelty_count):
                for k1 in range(num_pushing_contact_distance_count):
                    for l1 in range(num_pushing_delay_count):
                        Temp = video_label_accuracy[j1][i1][k1][l1]                   
                        total_count += 1
                        result[i1, j1, k1, l1] = Temp[3,:].mean()
        result *= 100.
        ##############################################################################################

        #plot based on delay and contact distance
        Temp_result = result.mean(axis=0).mean(axis=0)
        print('pushing results based on delay and distance: {}' .format(Temp_result))

        #plot based on delay  
        plt.title('Pushing result')
        for i in range(Temp_result.shape[0]):
            Temp_legend = 'contact distance: {}'.format(self.pushing_contact_distance[i]) # put the heuristics name here
            plt.plot(Temp_result[i,:], '-o', label=Temp_legend)

            plt.xticks(np.arange(Temp_result.shape[1]), self.pushing_delay)
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
        Temp_folder_name = '{}images/'.format(classification_data_save_path)
        Temp_folder_name = ss_folder_create(Temp_folder_name)
        Temp_file_name = '{0}tracked_pushing_result_based_on_delay_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
        plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

        plt.show(block=False)
        plt.close()
        ##############################################################################################

        #plot based on contact distance  
        plt.title('Pushing result')
        for i in range(Temp_result.shape[1]):
            Temp_legend = 'delay: {}'.format(self.pushing_delay[i]) # put the heuristics name here
            plt.plot(Temp_result[:,i], '-o', label=Temp_legend)

            plt.xticks(np.arange(Temp_result.shape[0]), self.pushing_contact_distance)
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
        Temp_folder_name = '{}images/'.format(classification_data_save_path)
        Temp_folder_name = ss_folder_create(Temp_folder_name)
        Temp_file_name = '{0}tracked_pushing_result_based_on_distance_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
        plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form

        plt.show(block=False)
        plt.close()
        ##############################################################################################

        final_result = result
        Temp_file_name = '{0}all_tracked_pushing_result_based_on_distance_nobj_{1}_mspeed_{2}_heurist_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  pushing_heuristics.replace(' ','_'), pushing_test_data_date)
        np.savez(Temp_file_name, final_result=final_result, video_label_accuracy=video_label_accuracy)
        ##############################################################################################  

        print('##############################################################################################')
    ##############################################################################################


