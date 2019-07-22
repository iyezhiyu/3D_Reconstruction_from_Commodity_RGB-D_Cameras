#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:59:11 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
This program aims to use the result from Mask RCNN and extracts the depth
information of each mask(instance).
"""
import os
import scipy.io as scio
import json
import imageio
import numpy as np
import time


if __name__ == "__main__":
    
    ycb_video_dir = '/Users/zhiyu/Desktop/YCB_Video_Dataset'
    extract_dir = '/Users/zhiyu/Desktop/extract_result'
    extract_annotation_dir = '/Users/zhiyu/Desktop/extract_result_annotations'
    out_dir = '/Users/zhiyu/Desktop/depth_instances'


    # The videos to process
    folder_names = os.listdir(extract_dir)
    videos = []
    for folder in folder_names:
        if os.path.isdir(extract_dir+'/' + folder):
            videos.append(folder)
    videos.sort()
    
    
    # Process for each video
    for video in videos:
        
        start = time.time()
        print("============================================")
        print("Processing video :", video)
        
        json_output = []
        video_out_dir = out_dir + '/' + video
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        with open(extract_annotation_dir + '/' + video + '/annotations.json','r') as load_f:
            annotations = json.load(load_f)
            
        for annotation in annotations:
            
            image_name = annotation['image_name'][-16:-10]
            
            boxes_t = annotation['boxes']
            classes_t = annotation['classes']
            segms_t =  annotation['segms']
            boxes = []
            classes = []
            segms = []
            for i in range(len(boxes_t)):
                if boxes_t[i][4] > 0.9: # define that the confidence should be > 0.9
                    boxes.append(boxes_t[i][:4])
                    classes.append(classes_t[i])
                    segms.append(segms_t[i])
            
            depth_image = imageio.imread(extract_dir + '/' + video + '/' + image_name + '-depth.png')
            mat_file = scio.loadmat(extract_dir + '/' + video + '/' + image_name + '-meta.mat')
            factor_depth = float(mat_file['factor_depth'][0][0])
            intrinsic_matrix = mat_file['intrinsic_matrix'].tolist()
            rotation_translation_matrix = mat_file['rotation_translation_matrix'].tolist()
            
            for i in range(len(classes)):
                new_depth_name = image_name + '_' + str(classes[i]) + '.png'
                segm = segms[i]
                new_depth_image = np.array(depth_image * segm, dtype = np.uint16)
                json_output.append({'image_name': new_depth_name, \
                                    'class': classes[i], \
                                    'factor_depth': factor_depth, \
                                    'intrinsic_matrix': intrinsic_matrix, \
                                    'rotation_translation_matrix': rotation_translation_matrix})
                imageio.imwrite(video_out_dir + '/' + new_depth_name, new_depth_image)
                
        with open(video_out_dir + '/annotations.json', 'w') as f:
            json.dump(json_output, f)
            
        print("Processing video", video, "done")
        print("Time used:", round(time.time() - start, 2))
        print("============================================")