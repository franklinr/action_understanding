###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 270418/050918
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%--------------------------------------------------------------------------------------
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython.display import HTML
##############################################################################################

import os
import sys

ss_lib_path = 'SS_PYLIBS/'
sys.path.insert(0, ss_lib_path)

#---------------------------------------------------------------------------------------------
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

                 
class ss_chasing(object):
    
    def __init__(self, args):
    
        self.todaysdate = time.strftime("%d%m%Y")#give the date in ddmmyyyy format

        ##parameters of the chasing action
        self.chasing_subtelty_angle = args.chasing_subtelty_angle
        if not isinstance(self.chasing_subtelty_angle, np.ndarray):# range of chasing subtelties
            self.chasing_subtelty_angle = [self.chasing_subtelty_angle]        
        self.Temp_max_speed = [args.max_speed]# speed of each velocity
        self.Temp_max_steer_force = [args.max_steer_force]#steering force corresponding to the object speed
        self.num_objects = args.chas_num_objects  #number of circle at each video
        self.object_radius = args.object_radius # object radius (As used by Gao)
        self.object_circumference_width = args.object_circumference_width #circular object circumference width (As used by Gao)
        #object_color = args.object_color[1:-1].replace(' ','').split(',')
        #self.object_color = np.reshape(np.array([int(object_color[0]), int(object_color[1]), int(object_color[2])]),(3, 1))#object color (As used by Gao)
        self.object_color = np.reshape(np.array([args.object_color, args.object_color, args.object_color]),(3, 1))#object color (As used by Gao)
        self.row = args.chas_row#video frame size (As used by Gao)
        self.column = args.chas_column#video frame size (As used by Gao)
        self.image_background = args.image_background#video frame background color
        self.tajectory_length = args.tajectory_length #number of frames in a video
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
        print('generating chasing data')
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        for stms in range(len(self.Temp_max_speed)):
            for scs in self.chasing_subtelty_angle:
                
                ########DIFFERENT PARAMETERS FOR DATA GENERATIONS###############
                dx_row, dx_column = 5*self.object_radius + 1, 5*self.object_radius + 1 #handling image boundaries
                dx, dy = 1.2*dx_row + 1, 1.2*dx_column + 1 # grid cells for 2d pont sample
                int_theta = 0.0
                end_theta = 360.0
                theta_div = 1.0
                replacement_flag = 0 # random sample replacement flag
                max_speed = self.Temp_max_speed[stms]*np.ones((1, self.num_objects)) # constant velocity of each objects(best 5.0)
                max_speed[0,0] += np.exp(-np.sqrt(np.sqrt(scs/20)))
                max_steer_force = self.Temp_max_steer_force[stms]*np.ones((1, self.num_objects)) # maximum steering force(best 3.5)
                # disjoint_percentage = 1 # between(0-overlap to 1-non-overlap); object overlap factor
                disjoint_threshold = 8.0*self.object_radius*np.ones((1, self.num_objects))#sp.reshape(int(disjoint_percentage*(3*(2*self.object_radius+1))+1)*np.ones((1, self.num_objects)),(1, self.num_objects))
                velo_theta_thresh = 60.0*np.ones((1, self.num_objects)) #velocity moving direction range from perious frame.
                velo_theta_thresh[0, 1] = scs #for wolf subtelty
    #             disjoint_threshold[0, 1] = 2*disjoint_threshold[0, 1]
                ################################################################

                for snv in range(0, self.num_video_files):
                    print('----------------------------------------------------------------------------------------------')
                    print('generating {}(/{})-th video data for subtlety {}' .format(snv, self.num_video_files, scs/2))
                    Temp_file_name_mask = 'cs_{}_mspeed_{}_{}' .format(int(scs/2.0), int(self.Temp_max_speed[stms]), Temp_file_number%snv)
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

                    #############TAKE WOLF MAX DISTANCE FROM SHEEP###########
                    Temp_dist = ss_euclidean_dist(np.reshape(int_pos[:,0], (2, 1)).copy(), int_pos.copy())
                    Temp_X = int_pos[:,1].copy()
                    int_pos[:,1] = int_pos[:,Temp_dist.argmax()].copy()
                    int_pos[:,Temp_dist.argmax()] = Temp_X.copy()
                    #########################################################


                    # call actual data generation api
                    if self.verbose_flag:
                        print('WORKING ACTUAL DATA GENERATING FUNCTION.....')
                    pos, pos_velo = ss_chasing_random_trajectory(int_pos, int_pos_velo, max_speed, max_steer_force, self.row, self.column, dx_row, dx_column, self.tajectory_length, velo_theta_thresh, disjoint_threshold)
                    
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

                    # for video file
                    if(self.video_saving_flag):
                        Temp_folder_name = '{}video/without_label/'.format(self.data_save_path)
                        Temp_file_name = 'video_file_{}'.format(Temp_file_name_mask)
                        ss_images2video(0, images, Temp_folder_name, Temp_file_name, self.ougput_video_file_extention)# for video without label info
                        
                        #for object label
                        if(self.video_saving_flag_label):
                            # put object label info
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
        print('testing chasing based on original position data')
        ##############################################################################################
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        pos_data_path = '{}pos/'.format(self.data_save_path)
        video_data_path = '{}video/without_label/'.format(self.data_save_path)
        classification_data_save_path = args.classification_data_save_path
        if not len(classification_data_save_path):
            classification_data_save_path = 'classification_results/{}/'.format(self.data_save_path)
        
        action_start_frame_id = args.chasing_action_start_frame_id
        heuristic_add_factor = args.chasing_heuristic_add_factor
        chasing_test_data_date = args.chasing_test_data_date
        if not len(chasing_test_data_date):
            chasing_test_data_date = self.todaysdate
        print('testing data taking from {}' .format(pos_data_path))
        org_chasing_info = np.array([[1],[0]])# original chasing class label
        video_label_accuracy = [None]*len(self.chasing_subtelty_angle)# for result save
        chasing_subtelty_count = 0
        ##############################################################################################
        
        for stms in range(len(self.Temp_max_speed)):
            for scs in self.chasing_subtelty_angle:
                video_label_accuracy[chasing_subtelty_count] = np.zeros((3, self.num_video_files))
                for snv in range(0, self.num_video_files):
                    
                    print('----------------------------------------------------------------------------------------------')
                    print('classifying {}(/{})-th video data for subtlety {}' .format(snv, self.num_video_files, scs/2))
                    Temp_file_name_mask = 'cs_{}_mspeed_{}_{}' .format(int(scs/2.0), int(self.Temp_max_speed[stms]), Temp_file_number%snv)
                    ##############################################################################################
                    # read original position info
                    Temp_file_name = '{}pos_{}.p'.format(pos_data_path, Temp_file_name_mask)
                    pos = ss_file_read(Temp_file_name)
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
                    
                    # call actual classification api
                    est_chasing_class_ids, est_chasing_info, est_chasing_score = ss_chasing_detection(pos, action_start_frame_id, heuristic_add_factor)
                    # match with the ground truth data
                    chasing_count, chasing_match_ids = ss_chasing_match(org_chasing_info, est_chasing_info)
                    #print(chasing_count, chasing_match_ids)
                    ##############################################################################################
                    
                    video_label_accuracy[chasing_subtelty_count][0, snv] = float(org_chasing_info.shape[1])
                    video_label_accuracy[chasing_subtelty_count][1, snv] = float(chasing_count)
                    avg_final_score = est_chasing_score[org_chasing_info[0,:],org_chasing_info[1,:],-1].mean()
                    if avg_final_score > 1.:
                        avg_final_score = 1.
                    video_label_accuracy[chasing_subtelty_count][2, snv] = avg_final_score
                    ##############################################################################################
                    
                    # save the results
                    Temp_folder_name = '{}chache/'.format(classification_data_save_path)
                    Temp_folder_name = ss_folder_create(Temp_folder_name)
                    Temp_file_name = '{}est_chas_info_{}'.format(Temp_folder_name, Temp_file_name_mask)
                    np.savez(Temp_file_name, est_chasing_class_ids=est_chasing_class_ids, est_chasing_info=est_chasing_info, est_chasing_score=est_chasing_score)
                    ##############################################################################################
                    
                    # add the cost matrix to the video 
                    num_frames = est_chasing_score.shape[2]
                    for i in xrange(num_frames):
                        cost_img = ss_image_create_cost_matrix(est_chasing_score[:,:,i], cost_image_size=[160, 160], display_flag='area', title='Chasing Score Map')
                        T_image = (255*np.ones((images[i].shape[0]-cost_img.shape[0], cost_img.shape[1], 3))).astype('uint8')
                        T_image = np.concatenate((cost_img, T_image), axis=0)
                        images[i] = np.concatenate((images[i], T_image), axis=1)
                    ##############################################################################################
                    
                    # save video file
                    if(self.video_saving_flag):
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
                    
                chasing_subtelty_count += 1
                
            ##############################################################################################
            
            #calculating accuracy and plot the results
            num_chasing_subtlety = len(video_label_accuracy)
            result = np.zeros((2, num_chasing_subtlety))
            for i in range(num_chasing_subtlety):
                result[0,i] = self.chasing_subtelty_angle[i]/2
                
                Temp = video_label_accuracy[i][1,:]/video_label_accuracy[i][0,:]
                if isinstance(Temp, np.ndarray):
                    Temp = Temp.mean()
                result[1,i] = 100.*Temp
                
            print('chasing classification accuracy(video wise): {}' .format(result))
            ##############################################################################################
            
            #plot and save the classification results 
            #read Gao result
            gao_result = (10./6)*np.array([53, 52, 27.5, 15, 21, 8])

            plt.title('Chasing result:({}-objects )'.format(self.num_objects))
            yticks_min = 10*(np.int(max(0, np.min(result[1,:])-20))/10)

            #plot Gao result
            Temp_legend = 'Gao et. al.'
            plt.plot(gao_result, '-o', label=Temp_legend)
            Temp_yticks_min = np.min(gao_result)
            if(yticks_min > Temp_yticks_min):
                yticks_min = Temp_yticks_min

            Temp_legend = 'Relative angle of motion'
            plt.plot(result[1,:-1], '-o', label=Temp_legend)

            plt.xticks(np.arange(result.shape[1]), (result[0,:-1].astype('int')))#('0', '30', '60', '90', '120', '150') )
            plt.xlabel('Chasing subtlety')

            yticks_max = 101
            yticks_interval = 5
            plt.yticks(np.arange(yticks_min, yticks_max, yticks_interval))
            plt.ylabel('Accuracy (%)')

            plt.grid(linestyle='--')
            plt.legend(loc='upper right')
            ##############################################################################################
            
            #save plot results
            Temp_folder_name = '{}images/'.format(classification_data_save_path)
            Temp_folder_name = ss_folder_create(Temp_folder_name)
            Temp_file_name = '{0}chasing_result_nobj_{1}_mspeed_{2}_heuristfact_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  round(heuristic_add_factor, self.decimal_file_number), chasing_test_data_date)
            plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
            plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form
            
            plt.show(block=False)
            plt.close()
            ##############################################################################################

            final_result = result
            Temp_file_name = '{0}all_chasing_result_nobj_{1}_mspeed_{2}_heuristfact_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  round(heuristic_add_factor, self.decimal_file_number), chasing_test_data_date)
            np.savez(Temp_file_name, final_result=final_result, video_label_accuracy=video_label_accuracy)
            ##############################################################################################    
                    
        print('##############################################################################################')                    

    def data_classification_track(self, args):
     
        print('##############################################################################################')
        print('testing chasing based on tracking data')
        ##############################################################################################
        Temp_file_number = '%'+str(self.decimal_file_number)+'.'+str(self.decimal_file_number)+'d'
        pos_data_path = '{}pos/'.format(self.data_save_path)
        video_data_path = '{}video/without_label/'.format(self.data_save_path)
        classification_data_save_path = args.classification_data_save_path
        if not len(classification_data_save_path):
            classification_data_save_path = 'classification_results/{}/'.format(self.data_save_path)
        classification_data_save_path = '{}tracked/'.format(classification_data_save_path)
        
        action_start_frame_id = args.chasing_action_start_frame_id
        heuristic_add_factor = args.chasing_heuristic_add_factor
        chasing_test_data_date = args.chasing_test_data_date
        if not len(chasing_test_data_date):
            chasing_test_data_date = self.todaysdate
        print('testing data taking from {}' .format(pos_data_path))
        org_chasing_info = np.array([[1],[0]])# original chasing class label
        video_label_accuracy = [None]*len(self.chasing_subtelty_angle)# for result save
        chasing_subtelty_count = 0
        ##############################################################################################
        
        for stms in range(len(self.Temp_max_speed)):
            for scs in self.chasing_subtelty_angle:
                video_label_accuracy[chasing_subtelty_count] = np.zeros((3, self.num_video_files))
                for snv in range(0, self.num_video_files):
                    
                    print('----------------------------------------------------------------------------------------------')
                    print('classifying {}(/{})-th video data for subtlety {}' .format(snv, self.num_video_files, scs/2))
                    Temp_file_name_mask = 'cs_{}_mspeed_{}_{}' .format(int(scs/2.0), int(self.Temp_max_speed[stms]), Temp_file_number%snv)
                    ##############################################################################################
                    # read original position info
                    Temp_file_name = '{}pos_{}.p'.format(pos_data_path, Temp_file_name_mask)
                    pos = ss_file_read(Temp_file_name)
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
                    
                    # call actual classification api
                    est_chasing_class_ids, est_chasing_info, est_chasing_score = ss_chasing_detection(tracked_pos, action_start_frame_id, heuristic_add_factor)
                    # match with the ground truth data
                    chasing_count, chasing_match_ids = ss_chasing_match(org_chasing_info, est_chasing_info)
                    #print(chasing_count, chasing_match_ids)
                    ##############################################################################################
                    
                    video_label_accuracy[chasing_subtelty_count][0, snv] = float(org_chasing_info.shape[1])
                    video_label_accuracy[chasing_subtelty_count][1, snv] = float(chasing_count)
                    avg_final_score = est_chasing_score[org_chasing_info[0,:],org_chasing_info[1,:],-1].mean()
                    if avg_final_score > 1.:
                        avg_final_score = 1.
                    video_label_accuracy[chasing_subtelty_count][2, snv] = avg_final_score
                    ##############################################################################################
                    
                    # save the results
                    Temp_folder_name = '{}chache/'.format(classification_data_save_path)
                    Temp_folder_name = ss_folder_create(Temp_folder_name)
                    Temp_file_name = '{}tracked_est_chas_info_{}'.format(Temp_folder_name, Temp_file_name_mask)
                    np.savez(Temp_file_name, est_chasing_class_ids=est_chasing_class_ids, est_chasing_info=est_chasing_info, est_chasing_score=est_chasing_score)
                    ##############################################################################################
                    
                    # add the cost matrix to the video 
                    num_frames = est_chasing_score.shape[2]
                    for i in xrange(num_frames):
                        cost_img = ss_image_create_cost_matrix(est_chasing_score[:,:,i], cost_image_size=[160, 160], display_flag='area', title='Chasing Score Map')
                        T_image = (255*np.ones((images[i].shape[0]-cost_img.shape[0], cost_img.shape[1], 3))).astype('uint8')
                        T_image = np.concatenate((cost_img, T_image), axis=0)
                        images[i] = np.concatenate((images[i], T_image), axis=1)
                    ##############################################################################################
                    
                    # save video file
                    if(self.video_saving_flag):
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
                    
                chasing_subtelty_count += 1
                
            ##############################################################################################
            
            #calculating accuracy and plot the results
            num_chasing_subtlety = len(video_label_accuracy)
            result = np.zeros((2, num_chasing_subtlety))
            for i in range(num_chasing_subtlety):
                result[0,i] = self.chasing_subtelty_angle[i]/2
                
                Temp = video_label_accuracy[i][1,:]/video_label_accuracy[i][0,:]
                if isinstance(Temp, np.ndarray):
                    Temp = Temp.mean()
                result[1,i] = 100.*Temp
                
            print('chasing classification accuracy(video wise): {}' .format(result))
            ##############################################################################################
            
            #plot and save the classification results 
            #read Gao result
            gao_result = (10./6)*np.array([53, 52, 27.5, 15, 21, 8])

            plt.title('Chasing result:({}-objects )'.format(self.num_objects))
            yticks_min = 10*(np.int(max(0, np.min(result[1,:])-20))/10)

            #plot Gao result
            Temp_legend = 'Gao et. al.'
            plt.plot(gao_result, '-o', label=Temp_legend)
            Temp_yticks_min = np.min(gao_result)
            if(yticks_min > Temp_yticks_min):
                yticks_min = Temp_yticks_min

            Temp_legend = 'Relative angle of motion'
            plt.plot(result[1,:-1], '-o', label=Temp_legend)

            plt.xticks(np.arange(result.shape[1]), (result[0,:-1].astype('int')))#('0', '30', '60', '90', '120', '150') )
            plt.xlabel('Chasing subtlety')

            yticks_max = 101
            yticks_interval = 5
            plt.yticks(np.arange(yticks_min, yticks_max, yticks_interval))
            plt.ylabel('Accuracy (%)')

            plt.grid(linestyle='--')
            plt.legend(loc='upper right')
            ##############################################################################################
            
            #save plot results
            Temp_folder_name = '{}images/'.format(classification_data_save_path)
            Temp_folder_name = ss_folder_create(Temp_folder_name)
            Temp_file_name = '{0}tracked_chasing_result_nobj_{1}_mspeed_{2}_heuristfact_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  round(heuristic_add_factor, self.decimal_file_number), chasing_test_data_date)
            plt.savefig('{}.eps'.format(Temp_file_name))#save the results in eps form
            plt.savefig('{}.png'.format(Temp_file_name))#save the results in jpg form
            
            plt.show(block=False)
            plt.close()
            ##############################################################################################

            final_result = result
            Temp_file_name = '{0}all_tracked_chasing_result_nobj_{1}_mspeed_{2}_heuristfact_{3}_date_{4}'.format(Temp_folder_name, self.num_objects, int(self.Temp_max_speed[stms]),  round(heuristic_add_factor, self.decimal_file_number), chasing_test_data_date)
            np.savez(Temp_file_name, final_result=final_result, video_label_accuracy=video_label_accuracy)
            ##############################################################################################    
                    
        print('##############################################################################################')                    





