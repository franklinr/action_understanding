
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from ss_computation import *

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_color_generation(num_color, sample_flag = 'uniform'):

    color_dims  = np.ceil(pow(num_color+5,(1.0/3)))
    color_dims = np.floor(255.0/color_dims)
    b_dims = np.arange(0, 255, color_dims)
    g_dims = np.arange(0, 255, color_dims)
    r_dims = np.arange(0, 255, color_dims)
    bv, gv, rv = np.meshgrid(b_dims, g_dims, r_dims)
    Temp_color = np.concatenate((np.reshape(bv,(1, bv.size)), np.reshape(gv,(1, gv.size)), np.reshape(rv,(1, rv.size)))) 
    Temp_color = Temp_color[:,2:-1]
    color_code, idx = ss_take_sample(Temp_color, sample_flag, num_color)
    color_code = color_code.astype(np.int)
    
    return(color_code)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_draw_lineOnImage(image, start_pos, end_pos, line_color = [], line_thickness = 1):
    
    start_pos = start_pos.astype(int)
    end_pos = end_pos.astype(int)
    num_lines = end_pos.shape[1]
    if len(line_color) == 0:
        line_color = ss_color_generation(num_lines)
        line_color = line_color.astype(int)
    else:
        if ((line_color.shape[1] == 1) & (line_color.shape[1] != num_lines)):
            line_color = line_color*np.ones((3, num_lines), np.int)
    if np.size(line_thickness) == 0:
        line_thickness = 2*np.ones((num_lines), np.int)
    elif np.size(line_thickness) == 1:
        line_thickness = line_thickness*np.ones((num_lines), np.int)
    else:
        if (np.size(line_thickness) != num_lines):
            line_thickness = line_thickness*np.ones((num_lines), np.int)
    for nln in range(num_lines):
        cv2.line(image, tuple(np.reshape(start_pos[:,nln],(2,1))), tuple(np.reshape(end_pos[:,nln],(2,1))), tuple(line_color[:, nln]), line_thickness[nln])
        
    return(image)  

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_draw_circlesOnImage(image, circle_center, circle_radius, circle_color = [], circumference_width = 3):
    
    circle_center = circle_center.astype(int)
    circle_radius = circle_radius.astype(int)
    num_circles = circle_center.shape[1]
    if len(circle_color) == 0:
        circle_color = ss_color_generation(num_circles)
        circle_color = circle_color.astype(int)
    else:
        if ((circle_color.shape[1] == 1) & (circle_color.shape[1] != num_circles)):
            circle_color = circle_color*np.ones((3, num_circles), np.int)
    if np.size(circumference_width) == 0:
        circumference_width = 3*np.ones((num_circles), np.int)
    elif np.size(circumference_width) == 1:
        circumference_width = circumference_width*np.ones((num_circles), np.int)
    else:
        if (np.size(circumference_width) != num_circles):
            circumference_width = circumference_width*np.ones((num_circles), np.int)
    for nc in range(num_circles):
        cv2.circle(image, tuple(np.reshape(circle_center[:,nc],(2,1))), circle_radius[nc], tuple(circle_color[:, nc]), circumference_width[nc])
        
    return(image)  

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_draw_textOnImage(image, text, text_pos, text_font_size = 2, text_color = [], text_width = 2, verbose_flag = 0):
    
    text_pos = text_pos.astype(int)
    num_text = np.size(text)
    if num_text!= text_pos.shape[1]:
        if verbose_flag:
            print('WARNING:NUMBER OF TEXT (%d) AND ITS POSITION (%d)IS NOT SAME \n'%(num_text, text_pos.shape[1]))
            print('WARNING:TAKING (%s) FOR ALL THE POSITIONS\n'%(text[0]))
        Temp_text = [None]*text_pos.shape[1]
        for sntx in range(text_pos.shape[1]):
            Temp_text[sntx] = text[0]
        text = Temp_text
        num_text = len(text)
    if np.size(text_font_size) == 0:
        text_font_size = 2*np.ones((num_text), np.int)
    elif np.size(text_font_size) == 1:
        text_font_size = text_font_size*np.ones((num_text), np.int)
    else:
        if (np.size(text_font_size) != num_text):
            text_font_size = text_font_size*np.ones((num_text), np.int)
    if len(text_color) == 0:
        text_color = ss_color_generation(num_text)
        text_color = text_color.astype(int)
    else:
        if ((text_color.shape[1] == 1) & (text_color.shape[1] != num_text)):
            text_color = text_color*np.ones((3, num_text), np.int)
    if np.size(text_width) == 0:
        text_width = 2*np.ones((num_text), np.int)
    elif np.size(text_width) == 1:
        text_width = text_width*np.ones((num_text), np.int)
    else:
        if (np.size(text_width) != num_text):
            text_width = text_width*np.ones((num_text), np.int)
    for ntxt in range(num_text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text[ntxt], tuple(np.reshape(text_pos[:,ntxt],(2,1))), font, text_font_size[ntxt], tuple(text_color[:, ntxt]), text_width[ntxt])
        
    return(image)

##############################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-16
#% For bug and others mail me at soumitramath39@gmail.com
#%---------------------------------------------------------------------------------------------
#% INPUT: 
#% OUTPUT: 
#%--------------------------------------------------------------------------------------
#% EXAMPLE: 
#%
##########################################################################
def ss_frame_generate(row, column, image_background, pos, circle_radius, circle_color, circumference_width):

    num_points = pos.shape[1]
    num_frames = pos.shape[2]

    video_frame = list(range(num_frames))
    for nframe in range(num_frames):
        Temp_pos = pos[...,nframe]
        Temp_pos = Temp_pos[[1, 0],:]
        temp_image = image_background*np.ones((row, column, 3), np.uint8)
        temp_image = ss_draw_circlesOnImage(temp_image, Temp_pos, circle_radius, circle_color, circumference_width)
        video_frame[nframe] = temp_image
        
    return(video_frame)

#############################################################################

def ss_plot_confusion_matrix(confusion_matrix, class_info=[], title='Confusion matrix', color_map=plt.cm.Reds, area_flag=0):
    
    num_class = confusion_matrix.shape[0]
    max_val = float(confusion_matrix.max())
    if len(class_info)==0:
        class_info = xrange(num_class)
    #plt.imshow(confusion_matrix, interpolation='nearest', cmap=color_map)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, class_info, rotation=45)
    plt.yticks(tick_marks, class_info)
    for i in tick_marks:
        plt.axvline(x=i+.5, color='black')
        plt.axhline(y=i+.5, color='black')
#         patches.Rectangle(xy, width, height, angle=0.0, **kwargs)
        if area_flag:
            max_val = 1
        
            for j in tick_marks:
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), np.sqrt(confusion_matrix[i,j]/max_val), np.sqrt(confusion_matrix[i,j]/max_val), fill=True, color='g'))
    
    plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

def ss_image_create_cost_matrix(cost_mat, cost_image_size=[200, 200], display_flag='color_map', label_flag=1, title=[], title_font_size=0.4, title_color=np.array([[255],[0],[0]]), title_width=1):
    
    row = cost_image_size[0]
    col = cost_image_size[1]
    image = 255.*np.ones((row, col, 3))
    max_val = cost_mat.max()
    if cost_mat.max() > 1.:
        cost_mat /= max_val
    
    if cost_mat.ndim == 2:
        if((cost_mat.shape[0] <= row) & (cost_mat.shape[1] <= col)):
            cell_size_row = np.int(np.floor((row-cost_mat.shape[0])/cost_mat.shape[0]))
            cell_size_col = np.int(np.floor(float(col-cost_mat.shape[1])/cost_mat.shape[1]))
            Temp_id_row = np.arange(0, row, cell_size_row+1)
            Temp_id_col = np.arange(0, col, cell_size_col+1)
            
            
            for i in range(cost_mat.shape[0]):
                if i:
                     image[Temp_id_row[i]-1,:,:]= 0
                for j in range(cost_mat.shape[1]):
                    if j:
                        image[:,Temp_id_col[j]-1,:]= 0
                    if display_flag=='color_map':
                        image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,1] = (image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,1] - 255.*(cost_mat[i, j]))
                        image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,2] = (image[Temp_id_row[i]:Temp_id_row[i]+cell_size_row, Temp_id_col[j]:Temp_id_col[j]+cell_size_col,2] - 255.*(cost_mat[i, j]))
                    if display_flag=='area':
                        Temp_r = int(np.round(cell_size_row*np.sqrt(cost_mat[i, j])))
                        Temp_c = int(np.round(cell_size_col*np.sqrt(cost_mat[i, j])))
                        rect_color = (255, 255*(1-cost_mat[i, j]), 255*(1-cost_mat[i, j]))
                        cv2.rectangle(image, (Temp_id_col[j],Temp_id_row[i]),(Temp_id_col[j]+Temp_c,Temp_id_row[i]+Temp_r),rect_color,-1)
            # for border 
            image[0,:,:]= 0
            image[-1,:,:]= 0
            image[:,0,:]= 0
            image[:,-1,:]= 0
            if label_flag:             
                # for cost matrix labels
                # for horizontal labels
                label_image_row = 255.*np.ones((cell_size_row, col, 3))
                Temp_i = int(cell_size_row/2.)
                for j in range(cost_mat.shape[1]):
                    Temp_j = int(Temp_id_col[j] + cell_size_col*0.3)
                    label_image_row = ss_draw_textOnImage(label_image_row, [str(j)], np.array(([[Temp_j],[Temp_i]])), title_font_size, title_color, title_width)
                image = np.concatenate((image, label_image_row), axis=0)   

                # for vertical labels
                label_image_col = 255.*np.ones((row, cell_size_col, 3))
                Temp_j = int(cell_size_col/2.)
                for i in range(cost_mat.shape[0]):
                    Temp_i = int(Temp_id_row[i] + cell_size_row*0.7)
                    label_image_col = ss_draw_textOnImage(label_image_col, [str(i)], np.array(([[Temp_j],[Temp_i]])), title_font_size, title_color, title_width)


                T_image = (255*np.ones((image.shape[0]-label_image_col.shape[0], label_image_col.shape[1], 3))).astype('uint8')
                label_image_col = np.concatenate((label_image_col, T_image), axis=0)
                image = np.concatenate((label_image_col, image), axis=1)   
            
            
            # for title
            if len(title):
                title_image = 255.*np.ones((cell_size_row, col, 3))
                title_image = ss_draw_textOnImage(title_image, [title], np.array(([[5],[int(cell_size_row/1.5)]])), title_font_size, title_color, title_width)
                T_image = (255*np.ones((title_image.shape[0], image.shape[1]-title_image.shape[1], 3))).astype('uint8')
                title_image = np.concatenate((T_image, title_image), axis=1)
                image = np.concatenate((title_image, image), axis=0)
        else:
           
            raise ErrorValue('TARGET IMAGE (%d, %d) SMALLER THAN SOURCE IMAGE (%d, %d)' %(row, col, cost_mat.shape[0], cost_mat.shape[1]))
    
    image = image.astype('uint8')
    
    return image


