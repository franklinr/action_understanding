###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 260418/050918
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
import argparse
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
from ss_chasing_classify_final import ss_chasing
from ss_pushing_classify_final import ss_pushing
##############################################################################################

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2
##############################################################################################

input_parser = argparse.ArgumentParser(description='')
input_parser.add_argument('--action_name', dest='action_name', default='pushing', help='name of the action(chasing or pushing; default: "pushing")')
# for chasing action
input_parser.add_argument('--chasing_subtelty_angle', dest='chasing_subtelty_angle', type=float, default=np.arange(0.0, 361.0, 60), help='range of chasing subtelties(default: "[0., 60., 120., 180., 240., 300., 360.]")')
input_parser.add_argument('--max_speed', dest='max_speed', type=float, default=5., help='avg. speed of each object(default: "5.")')
input_parser.add_argument('--max_steer_force', dest='max_steer_force', type=float, default=2., help='steering force of each object(default: "2.")')
input_parser.add_argument('--chas_num_objects', dest='chas_num_objects', type=int, default=4, help='no. of objects participate in action (default: "4")')
input_parser.add_argument('--object_radius', dest='object_radius', type=int, default=10, help='object radius (as used by Gao; default: "10") ')
input_parser.add_argument('--object_circumference_width', dest='object_circumference_width', type=int, default=2, help='circular object circumference width (As used by Gao; default: "2")')
input_parser.add_argument('--object_color', dest='object_color', type=int, default=255, help='object color (as used by Gao; default: "255") ')
input_parser.add_argument('--chas_row', dest='chas_row', type=int, default=480, help='video frame size in row(as used by Gao; default: "480")')
input_parser.add_argument('--chas_column', dest='chas_column', type=int, default=640, help='video frame size in column(as used by Gao; default: "640")')
input_parser.add_argument('--image_background', dest='image_background', type=int, default=0, help='video frame background color (0-black and 255-while; default: "0")')
input_parser.add_argument('--tajectory_length', dest='tajectory_length', type=int, default=500, help='number of frames in a video (default: "500")')
input_parser.add_argument('--num_video_files', dest='num_video_files', type=int, default=1, help='how many video files want to generate (default: "1")')
input_parser.add_argument('--object_init_reuse_flag', dest='object_init_reuse_flag', type=bool, default=False, help='initial object position taken from save position (default: "False") ')
input_parser.add_argument('--object_init_reuse_folder', dest='object_init_reuse_folder', type=str, default='', help='initial object position taken folder (default: "data_save_path)") ')


# for pushing action
input_parser.add_argument('--pushing_subtelty_angle', dest='pushing_subtelty_angle', type=float, default=np.arange(0.0, 181.0, 60), help='range of pushing subtelties(default: "[0., 60., 120., 181.]")')
input_parser.add_argument('--num_pushing', dest='num_pushing', type=int, default=np.array([1, 2, 3]), help='different number of pushing in a video(default: "[1, 2, 3]")')
input_parser.add_argument('--pushing_contact_distance', dest='pushing_contact_distance', type=int, default=np.array([0, 10, 20]), help='distance (in pixels) between pusher and pushe dusing pushing(default: "[0, 10, 20]")')
input_parser.add_argument('--pushing_delay', dest='pushing_delay', type=int, default=np.array([0, 10, 20]), help='pushing delay(in frames) in number of frames(default: "[0, 10, 20]")')
input_parser.add_argument('--push_interval', dest='push_interval', type=int, default=30, help='gap(in frames) between two pushing(default: "30")')
input_parser.add_argument('--push_num_objects', dest='push_num_objects', type=int, default=9, help='no. of objects participate in pushing action (default: "9")')
input_parser.add_argument('--push_row', dest='push_row', type=int, default=720, help='video frame size in row(as used by Andrew; default: "720")')
input_parser.add_argument('--push_column', dest='push_column', type=int, default=1280, help='video frame size in column(as used by Andrew; default: "1280")')

#output options
input_parser.add_argument('--position_saving_flag', dest='position_saving_flag', type=bool, default=True, help='object position save flag (0-no save, 1-save; default: "True")')
input_parser.add_argument('--video_saving_flag', dest='video_saving_flag', type=bool, default=True, help='video save flag (default: "True")')
input_parser.add_argument('--video_saving_flag_label', dest='video_saving_flag_label', type=bool, default=False, help='Video save flag with objects label(default: "False")')
input_parser.add_argument('--ougput_video_file_extention', dest='ougput_video_file_extention', type=str, default='mp4', help='video file format(default: "mp4")')
input_parser.add_argument('--data_save_path', dest='data_save_path', type=str, default='', help='data save path (position and video; default: "data/todays_date/")')
input_parser.add_argument('--decimal_file_number', dest='decimal_file_number', type=int, default=5, help='ouput file number in (decimal default: "5")')
input_parser.add_argument('--verbose_flag', dest='verbose_flag', type=bool, default=False, help='vervose flag (default: "False")')
 
# for classification
input_parser.add_argument('--classification_data_save_path', dest='classification_data_save_path', type=str, default='', help='action classification result save path (default: "classification_results/data_save_path/")')
# for chasing
input_parser.add_argument('--chasing_action_start_frame_id', dest='chasing_action_start_frame_id', type=int, default=5, help='action starting frame number (default: "5")')
input_parser.add_argument('--chasing_heuristic_add_factor', dest='chasing_heuristic_add_factor', type=float, default=0.97, help='action cost mat addition factor (default: "0.97")')
input_parser.add_argument('--chasing_test_data_date', dest='chasing_test_data_date', type=str, default='', help='chasing test data date (default: "todays_date")')

# for pushing
input_parser.add_argument('--pushing_action_start_frame_id', dest='pushing_action_start_frame_id', type=int, default=5, help='action starting frame number (default: "5")')
input_parser.add_argument('--pushing_frame_diff_thresh', dest='pushing_frame_diff_thresh', type=int, default=10, help='pushing action frame difference (default: "10")')
input_parser.add_argument('--pushing_avg_speed_frame_threshold', dest='pushing_avg_speed_frame_threshold', type=int, default=30, help='pushing average speed frame threshold   (default: "30")')
input_parser.add_argument('--pushing_heuristics', dest='pushing_heuristics', type=str, default='avg velocity before and after touch', help='pushing heuristics   (default: "avg velocity before and after touch")')
input_parser.add_argument('--pushing_test_data_date', dest='pushing_test_data_date', type=str, default='', help='pushing test data date (default: "todays_date")')



args = input_parser.parse_args()

def main():
    if args.action_name == 'chasing':
        print('classifying "{}" action data' .format(args.action_name))
        chasing = ss_chasing(args)# initialize the chasing action data generation
        chasing.data_generation()# generate the chasing action data
        chasing.data_classification_org(args)# classification based on original position information
        chasing.data_classification_track(args)# classification based on tracking position information
    elif args.action_name == 'pushing':
        print('classifying "{}" action data' .format(args.action_name))
        pushing = ss_pushing(args)# initialize the pushing action data generation
        pushing.data_generation()# generate the pushing action data
        pushing.data_classification_org(args)# classification based on original position information
        pushing.data_classification_track(args)# classification based on tracking position information
    else:
        print('define api for "{}" action' .format(args.action_name))

              

if __name__ == "__main__":
    main()
    
    
    
