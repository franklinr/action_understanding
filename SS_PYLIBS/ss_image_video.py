
import numpy as np
import glob
import os
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import ndimage
import skimage
import skimage.transform
from skimage import measure
import skvideo.io
from scipy.misc import imread, imsave, imresize

from IPython.display import clear_output

from ss_computation import*
from ss_input_output import *
##############################################################################################


########################################################################
#% FUNCTION: Read a set of images in a specified folder,specified file extention
#% WRITER: SOUMITRA SAMANTA            DATE: 03-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------------------
#% INPUT: data_path, file_extention: data_path- From where you want to read the images;
#%        file_extention- image file extension
#% OUTPUT: images:images- images(images[0], images[1],...) within the
#%      specified folder with specified file extension.
#%--------------------------------------------------------------------------------------
#% EXAMPLE:
#######################################################################
def ss_image_read(file_name):#FOR SINGLE IMAGE

    image = cv2.imread(file_name)
    if image.ndim == 3:
        image = image[:,:,[2, 1, 0]].copy()#21-06-17
        
    return(image)

def ss_images_read(data_path, file_extention = 'jpg', disp_flag = 0):#FOR MULTIPLE IMAGES IN A FOLDER

    if not data_path:
        print('ERROR: UNKNOWN IMAGE DATPATH " %s"!\n' % (data_path))
        return([])
    if not file_extention:
        print('ERROR: UNKNOWN IMAGE FILE EXTENSION ".%s"!\n' % (file_extention))
        return

    file_info = glob.glob(data_path+'*.'+file_extention)
    file_info = sorted(file_info)
    num_images = len(file_info)
    if num_images:
        images = list(range(num_images))
        for nimg in range(num_images):

            images[nimg] = cv2.imread(file_info[nimg])
            if images[nimg].ndim == 3:
                images[nimg] = images[nimg][:,:,[2, 1, 0]].copy()#21-06-17
            if disp_flag:
                print('READING "%s" FILE\n' %file_info[nimg])

    else:
        print('ERROR: NO IMAGES WERE FOUND WITH THE FOLDER "%s" WITH EXTENSION "%s"!\n' % (data_path, file_extention))
        return

    #return(images)
    return images, file_info #26-06-18

#######################################################################
#% FUNCTION: Showing image in current window
#% WRITER: SOUMITRA SAMANTA            DATE: 06-11-16
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------------------
#% INPUT: image: image- image which want to show
#% OUTPUT: .
#%--------------------------------------------------------------------------------------
#% EXAMPLE:
#######################################################################
def ss_images_show(images):

    num_images = len(images)
    #print(num_images)
    for nim in range(num_images):
        #print(nim)
        cv2.imshow('frames',images[nim])
        #if cv2.waitKey(1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return(1)

# def ss_images_show(images, title_name = [], img_hx = 10.0, img_hy = 8.0, ):
#
#     num_images = len(images)
#     plt.rcParams['figure.figsize'] = (img_hx, img_hy) # set default size of plots
#     for nim in range(num_images):
#         plt.axis('off')
#         if not title_name:
#             title_name = 'frame#-'+str(nim)
#         plt.title(title_name)
#         plt.imshow(images[nim])
#         plt.show()
#         clear_output(wait=True)
#     return(1)

#######################################################################
#% FUNCTION: Create an animator for showing video in jupyter notebook
#% WRITER: SOUMITRA SAMANTA            DATE: 06-11-17
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------------------
#% INPUT: image: image- image which want to show
#% OUTPUT: .
#%--------------------------------------------------------------------------------------
#% EXAMPLE: ani = ss_create_video_animation(images)
#                    HTML(ani.to_html5_video())
#######################################################################
def ss_create_video_animation(images, frame_interval=50, repeat_flag=False, blit_flag=True):
    
    fig = plt.figure()

    number_frames = len(images)
    Temp_images = [[None]]*number_frames
    for i in range(number_frames):
        im = plt.imshow(images[i], animated=True)
        Temp_images[i] = [im]

    ani_video = animation.ArtistAnimation(fig, Temp_images, interval=frame_interval, repeat=repeat_flag, blit=blit_flag)
    plt.close()
    return ani_video

#######################################################################
#% FUNCTION: Resize a set of images with a specified resize factor
#% WRITER: SOUMITRA SAMANTA            DATE: 21-05-16/260318
#% For bug and others mail me at soumitramath39@gmail.com
#%----------------------------------------------------------------------------------------
#% INPUT: images, resize_factor: images- image array which contains images
#%      (images[0], images[1],....); resize_factor- how much you
#%        want to resize the input images (default 0.5).
#% OUTPUT: resize_images: resize_images- resize images (resize_images[0],
#%         resize_images[1],...)of input images
#%--------------------------------------------------------------------------------------
#% EXAMPLE: resize_images = ss_image_resize(images, resize_factor)
#######################################################################
def ss_images_resize(images, resize_scale=[], resize_size=[]):

    num_images = len(images)
    resize_images = list(range(num_images))
    if((resize_scale) or (resize_size)):
        for nim in range(num_images):
            if resize_scale:
                resize_size =(np.floor(images[nim].shape[0]*resize_scale), np.floor(images[nim].shape[1]*resize_scale))
            resize_images[nim] = (255*skimage.transform.resize(images[nim], resize_size, mode='constant')).astype('uint8')
    else:
        raise ValueError('Please enter the resize scale between (0 1) as "resize_scale=?" \n or resize size as a tuple as "resize_size=(?,?)"\n' )
    return(resize_images)

#######################################################################
# % FUNCTION: Write a set of images in a specified folder,specified file with extention
# % WRITER: SOUMITRA SAMANTA            DATE: 05-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: images, save_path, file_name, file_extention, file_number: images-
# %        image array which contains images (images[0], images[1],....);
#%         save_path- Where you want to save the
# %        images(default Temp_save_image in current folder);
#%         file_name- images file name(default img);
# %        file_extention- image file extension(default jpg); file_number- file
# %        number decimal extension(default 5).
# % OUTPUT: 1
# %--------------------------------------------------------------------------------------
# % EXAMPLE: ss_image_write(images, 'Temp_save_image', 'img','jpg',3)
#######################################################################
def ss_images_write(images, save_path = 'Temp_image_save/', file_name = 'img', file_extention = 'jpg', file_number = 0, decimal_file_number = 5):

    save_path = ss_folder_create(save_path)
    num_images = len(images)
    if(num_images > 1):
        for nim in range(num_images):
            Temp_image = images[nim].astype('uint8')
            if(np.ndim(Temp_image) == 3):
                Temp_image = Temp_image[:,:,[2, 1, 0]];#21-06-17
            Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
            cv2.imwrite(save_path+file_name+Temp_file_number%(nim+file_number)+'.'+file_extention, Temp_image)#21-06-17
    else:
        Temp_image = images[0].astype('uint8')
        if(np.ndim(Temp_image) == 3):
            Temp_image = Temp_image[:,:,[2, 1, 0]];#21-06-17
        Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
        cv2.imwrite(save_path+file_name+Temp_file_number%file_number+'.'+file_extention, Temp_image)#21-06-17

    return(1)


#######################################################################
# % FUNCTION: Convert a set of rgb images into a gray images
# % WRITER: SOUMITRA SAMANTA            DATE: 05-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: images, save_path: images- image array which contains images
#%         (images[0], images[1],....); save_path-path where you want to save
# %       the coverted images.
# % OUTPUT: gray_images: gray_images- transformed gray images((gray_images[0], gray_images[1],....).
# %--------------------------------------------------------------------------------------
# % EXAMPLE: gray_images = ss_image_rgb2gray(images)
#######################################################################
def ss_images_rgb2gray(images, save_path = []):

    num_images = len(images)
    gray_images = list(range(num_images))
    for nim in range(num_images):
        if np.ndim(images[nim]) == 3:
            gray_images[nim] = cv2.cvtColor(images[nim].astype('uint8'), cv2.COLOR_RGB2GRAY)#21-06-17
        else:
            gray_images[nim] = images[nim]

    if save_path:
        ss_images_write(gray_images, save_path)

    return(gray_images)

#######################################################################
# % FUNCTION: Read a video file and save the images in a specified folder
# % WRITER: SOUMITRA SAMANTA            DATE: 05-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: input_file, save_path: input_file- input video file; save_path-
# %       folder where the images will be stored.
# % OUTPUT: images: images- images within the psecified video file (images[0], images[1],...)
#%------------------------------------------------------------------------------
#%  NOTE: Maximum numbe of frames 100000 (please change accordingly
# %--------------------------------------------------------------------------------------
# % EXAMPLE: images = ss_video2image('test.mp4', 'test/')
#######################################################################
# def ss_video2images(input_file, int_frame = [], end_frame = [], save_path = []):

#     cap = cv2.VideoCapture(input_file)
#     if int_frame:
#         cap.set(1, int_frame)
#     count = 0
#     images = list(range(100000))
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret==True:
#             images[count] = frame[:,:,[2, 1, 0]].copy()#21-06-17
#             count+=1
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#         # Release everything if job is finished
#     cap.release()
#     if save_path:
#         ss_images_write(images, save_path)

#     return(images[:count])
def ss_video2images(input_file=[], int_frame_pos_flag='index', int_frame_pos=[], end_frame_pos=[], save_path=[]):
    if os.path.isfile(input_file):
        vidcap = cv2.VideoCapture(input_file) # call video reader
        frame_rate = vidcap.get(5) # frame rate in the video file

        if int_frame_pos_flag=='index': # read video based on specified frame index information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = vidcap.get(7)
            num_frames = int(end_frame_pos - int_frame_pos)#.astype(int)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, int_frame_pos)

        else: # read video based on specified time (millisecond) information
            if not int_frame_pos:
                int_frame_pos = 0
            if not end_frame_pos:
                end_frame_pos = (1000.*vidcap.get(7))/frame_rate
            num_frames = np.floor(frame_rate*((end_frame_pos-int_frame_pos)/1000.)).astype(int)
            vidcap.set(cv2.CAP_PROP_POS_MSEC, int_frame_pos)

        images = list(range(num_frames)) # for video frame save
        count = 0
        while((vidcap.isOpened()) and (count<num_frames)) :
            ret, frame = vidcap.read()
            if ret==True:
                images[count] = frame[:,:,[2, 1, 0]].copy() # convert into RGB form 21-06-17
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        vidcap.release() # Release everything if job is finished
        if save_path:
            ss_images_write(images, save_path)

        return images
    else:
        raise ValueError('No such file {} does not exits!!' .format(input_file))

#######################################################################

def ss_images2video(disp_flag=0, images=[], save_folder = 'Temp_video_save/', output_file = 'Temp_video_file', output_file_format = 'avi', compression_method = 'mp4v', frames_second = 30, frame_size = []):

    num_images = len(images)
    video_save_file = save_folder+output_file+'.'+output_file_format
    save_folder = ss_folder_create(save_folder)

    if not frame_size:
        frame_size = images[0].shape
    #fourcc = cv2.cv.CV_FOURCC(*compression_method)#FOR OLDER VERSION OF OPENCV
    fourcc = cv2.VideoWriter_fourcc(*compression_method) #FOR NEW VERSION OF OPENCV
    out = cv2.VideoWriter(video_save_file, fourcc, frames_second, (frame_size[1],frame_size[0]))
    for nim in range(num_images):
        img = images[nim][:,:,[2, 1, 0]]#21-06-17
        out.write(img)
        if disp_flag:
            cv2.imshow('frame',img)
    out.release()
    cv2.destroyAllWindows()

    return(1)

#######################################################################

def ss_gray_threshold(image, thresh_method = 'hist_max'):

    if np.ndim(image) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#21-06-17
    if thresh_method == 'hist_max':
        img_hist = cv2.calcHist([image],[0],None,[256],[0,256])
        max_id = np.argmax(img_hist)
        gray_threshold = max_id
    else:
        print('ERROR: UNKNOWN GRAY IMAGE TRANSFORMATION THRESHOLD METHOD "%s"\n' %thresh_method)
        return 0

    return(gray_threshold)

#######################################################################
# % FUNCTION: Convert a rgb/gray images into binary images
# % WRITER: SOUMITRA SAMANTA            DATE: 06-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: image, thresh_method: image- image array which contains an image RGB/GRAY;
#%         thresh_method- binary threshold value calculation method (default 'otsu');
# % OUTPUT: bin_image, gray_thresh: binary_images- transformed binary image; gray_thresh-
#%          graylebel threshold value
# %--------------------------------------------------------------------------------------
# % EXAMPLE: bin_image, gray_thresh = ss_rgb_gray2binary(images, thresh_method)
#######################################################################
def ss_rgb_gray2binary(image, thresh_method = 'otsu', disp_flag = 0):

    if thresh_method == 'otsu':
        if np.ndim(image) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#21-06-17
        gray_thresh, bin_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif thresh_method == 'hist_max':
        if np.ndim(image) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#21-06-17
        gray_thresh = ss_gray_threshold(gray_image, thresh_method)
        bin_image = np.ones(gray_image.shape)
        bin_image[gray_image < gray_thresh +1] = 0
        gray_thresh = gray_thresh + 1
    elif (thresh_method == 'hist_max_channel') & (np.ndim(image) == 3):
        level1 = ss_gray_threshold(image[:,:,0])
        level2 = ss_gray_threshold(image[:,:,1]);
        level3 = ss_gray_threshold(image[:,:,2]);
        level = np.reshape([level1, level2, level3], (1, 1, 3))
        image = image - level
        image = np.sum(np.abs(image), axis=2)/3
        bin_image = np.ones(image.shape)
        bin_image[image < 4] = 0
        gray_thresh = np.round((level1 + level2 + level3)/3) + 4

    else:
        print('ERROR: UNKNOWN GRAY IMAGE TRANSFORMATION THRESHOLD METHOD "%s"\n' %thresh_method)
        return ([],[])
    bin_image = np.bool_(bin_image)
    if disp_flag:
        plt.imshow(bin_image)
        plt.show()

    return(bin_image, gray_thresh)

#######################################################################
#% FUNCTION: Convert a set of rgb/gray images into binary images
# % WRITER: SOUMITRA SAMANTA            DATE: 06-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: images, thresh_method, save_path: images- image array which contains images
#%         (images[0], images[1],...); thresh_method- binary threshold value calculation method (default 'otsu');
# %       save_path- path where you want to save the coverted images.
# % OUTPUT: bin_images, bin_thresh: bin_images- transformed binary images(bin_images[0], bin_images[1],....
#%         ; bin_thresh- binary threshold (bin_thresh[0], bin_thresh[1],.....)
# %--------------------------------------------------------------------------------------
# % EXAMPLE: (bin_images, bin_thresh) = ss_image_rgb_gray2binary(images, thresh_method,
# % save_path);
#######################################################################
def ss_images_rgb_gray2binary(images, thresh_method = 'otsu', save_path = []):

    num_images = len(images)
    bin_images = [None] *num_images
    bin_thresh = [None] *num_images
    for nim in range(num_images):
        (bin_image, gray_thresh) = ss_rgb_gray2binary(images[nim], thresh_method)
        if len(bin_image):
            bin_images[nim] = bin_image
            bin_thresh[nim] = gray_thresh
    if (len(save_path) > 0) & (bin_images[0] != 'None'):
        ss_images_write(bin_images, save_path)

    return(bin_images, bin_thresh)

#######################################################################
# % FUNCTION: Image linear stretching
# % WRITER: SOUMITRA SAMANTA            DATE: 06-11-16
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT: image, stretch_interval: image- image to be transformed; stretch_interval- transform scale
# % OUTPUT: stretch_images: stretch_images- transformed image
# %--------------------------------------------------------------------------------------
# % EXAMPLE: stretch_images = ss_image_linear_stretch(image, stretch_interval)
#######################################################################
def ss_image_linear_stretch(image, stretch_interval):

    if np.ndim(image) == 3:
        image = ss_images_rgb2gray([image])
        image = np.double(image[0])
    if not stretch_interval:
        stretch_interval = [np.min(image), np.max(image)]
    stretch_images = np.uint8(255*((image - stretch_interval[0])/ss_denominator_check(stretch_interval[1] - stretch_interval[0])))
    stretch_images = stretch_images.astype('uint8')

    return(stretch_images)

#######################################################################
# % FUNCTION:
# % WRITER: SOUMITRA SAMANTA            DATE:
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT:
# % OUTPUT:
# %--------------------------------------------------------------------------------------
# % EXAMPLE:
#######################################################################
def ss_area_threshold(bin_image, area_thresh):
    res_image = copy.deepcopy(bin_image)
    stats, num_pixels = ss_connected_components(bin_image)
    num_components = len(num_pixels)
    #$print('TEST %f %f\n' %(area_thresh, np.min(num_pixels)))
    for ncpt in range(num_components):
        if (num_pixels[ncpt] < area_thresh):
            pos = getattr(stats[ncpt], 'coords')
            res_image[pos[:, 0], pos[:, 1]] = 0
            #num_cord = pos.shape[0]
            #for snc in range(num_cord):
                #res_image[pos[snc, 0], pos[snc, 1]] = 0

    return(res_image)

#######################################################################
# % FUNCTION:
# % WRITER: SOUMITRA SAMANTA            DATE:
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT:
# % OUTPUT:
# %--------------------------------------------------------------------------------------
# % EXAMPLE:
#######################################################################
def ss_connected_components(bin_image):
    if bin_image.ndim > 2:
        bin_image, thres_val = ss_rgb_gray2binary(bin_image)
    conet_components_label, num_components = ndimage.label(bin_image, np.ones((3,3)))
    properties = measure.regionprops(conet_components_label)
    num_pixels = ss_list_with_attribute2array(properties, 'area')

    return(properties, num_pixels)

#######################################################################
# % FUNCTION:
# % WRITER: SOUMITRA SAMANTA            DATE:
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT:
# % OUTPUT:
# %--------------------------------------------------------------------------------------
# % EXAMPLE:
#######################################################################
def ss_backgound_estimation(image, thresh_method = 'hist_max_channel', disp_flag = 0):
    bin_image, level = ss_rgb_gray2binary(image, thresh_method)
    conect_properties, num_components = ss_connected_components(1-bin_image)
    max_id = np.argmax(num_components)
    image = np.double(image)
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    R_T = np.round(np.mean(R[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]))*np.ones(bin_image.shape)
    G_T = np.round(np.mean(G[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]))*np.ones(bin_image.shape)
    B_T = np.round(np.mean(B[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]))*np.ones(bin_image.shape)

    R_T[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]] = R[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]
    G_T[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]] = G[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]
    B_T[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]] = B[conect_properties[max_id].coords[:,0], conect_properties[max_id].coords[:,1]]
    background = np.stack((R_T, G_T, B_T), axis=2)
    if disp_flag:
        plt.imshow(background)
        plt.axis('off')
        plt.show()

    return(background)

#######################################################################
# % FUNCTION:
# % WRITER: SOUMITRA SAMANTA            DATE:
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT:
# % OUTPUT:
# %--------------------------------------------------------------------------------------
# % EXAMPLE:
#######################################################################
def ss_backgound_substruction(image, background = np.array([]), disp_flag = 0):

    if not background.size:
        background = ss_backgound_estimation(image)
    background_subs = np.abs(np.double(image) - np.double(background))
    background_subs = np.sum(background_subs, axis=2)
    foreground = np.zeros(background_subs.shape)
    foreground[background_subs > 0.1] = 1
    if disp_flag:
        plt.imshow(foreground,'gray')
        plt.axis('off')
        plt.show()

    return(foreground)

#######################################################################
# % FUNCTION:
# % WRITER: SOUMITRA SAMANTA            DATE: 31-05-17
# % For bug and others mail me at soumitramath39@gmail.com
# %----------------------------------------------------------------------------------------
# % INPUT:
# % OUTPUT:
# %--------------------------------------------------------------------------------------
# % EXAMPLE:
#######################################################################
def ss_take_image_rectangle(pos, dist_dx, dist_dy, row, column, dx_row=0, dx_column=0):

    int_r = pos[0,0] - dist_dx
    end_r = pos[0,0] + dist_dx
    int_c = pos[1,0] - dist_dy
    end_c = pos[1,0] + dist_dy
    row_low = dx_row
    row_up = row - dx_row - 1
    column_low = dx_column
    column_up = column - dx_column - 1
    if int_r <= row_low:
        int_r = row_low
        end_r = row_low + 2*dist_dx + 1
    if end_r >= row_up:
        end_r = row_up
        int_r = row_up - 2*dist_dx - 1
    if int_c <= column_low:
        int_c = column_low
        end_c = column_low + 2*dist_dy + 1
    if end_c >= column_up:
        end_c = column_up
        int_c = column_up - 2*dist_dy - 1
    return(int_r, end_r, int_c, end_c)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 260517
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_psnr_image(org_image, test_image):
    
    diff = org_image.astype('float32') - test_image.astype('float32')
    mse = np.mean(diff*diff)
    psnr = 10.*np.log10((255.*255)/mse)
    
    return(psnr)

###########################################################################
#% FUNCTION:
#% WRITER: SOUMITRA SAMANTA            DATE: 260318
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%---------------------------------------------------------------------------
#% EXAMPLE:
#%
##########################################################################
def ss_optical_flow(images, colType=0):
    
    #---------------------------------------------------------------------------------------------
    # Flow parameters:(default)
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    #---------------------------------------------------------------------------------------------
    
    num_frames = len(images)
    row, col = images[0].shape[0], images[0].shape[1]
    flow_x = np.zeros((row, col, num_frames))
    flow_y = np.zeros((row, col, num_frames))
    #---------------------------------------------------------------------------------------------
    for snf in range(num_frames-1):
        if colType: # for gray scale frame
            im1 = np.reshape(images[snf],(row, col, 1))
            im2 = np.reshape(images[snf+1],(row, col, 1))
        else: # for color frame
            im1 = images[snf]
            im2 = images[snf+1]
            
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
        flow_x[:,:,snf+1], flow_y[:,:,snf+1], im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
    #---------------------------------------------------------------------------------------------
    return flow_x, flow_y
