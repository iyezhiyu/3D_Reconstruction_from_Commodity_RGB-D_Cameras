#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 23:24:07 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
This program aims to generate the 3D point clouds of each instance.
"""
import os
import json
import time
import imageio
import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_points(video):
    
    start = time.time()
    print("============================================")
    print("Processing video :", video)
    print("Begin")
    
    video_dir = input_dir + '/' + video
    with open(video_dir + '/annotations.json','r') as load_f:
        annotations = json.load(load_f)
    
    for annotation in annotations:
        
        image_name = annotation['image_name']
        
        print("Processing", image_name)
        
        class_id = annotation['class']
        
        intrinsic_matrix = np.array(annotation['intrinsic_matrix'])
        fx = intrinsic_matrix[0][0]
        fy = intrinsic_matrix[1][1]
        cx = intrinsic_matrix[0][2]
        cy = intrinsic_matrix[1][2]
        
        # use np.linalg.inv to transfer T(o2c) to T(c2o)
        rotation_translation_matrix = np.linalg.inv(np.vstack((np.array(annotation['rotation_translation_matrix']), np.array([0,0,0,1]))))
        rotation_matrix = rotation_translation_matrix[:3,:3]
        translation_vector = np.reshape(rotation_translation_matrix[:3,3:4],(3,1))
        
        factor_depth = annotation['factor_depth']
        
        depth_image = imageio.imread(video_dir + '/' + image_name)/factor_depth
        
        min_,max_ = np.percentile(depth_image,[5,99])
        print(min_,max_)
        points_collection = []
        for v in range(h):
            for u in range(w):
                d = depth_image[v][u]
                if d == 0 or d > max_:
                    continue
                point_2 = d
                point_0 = (u - cx) * point_2 / fx
                point_1 = (v - cy) * point_2 / fy
                point = np.reshape(np.array([[point_0], [point_1], [point_2]]), (3,1))
                point_world = rotation_matrix.dot(point) + translation_vector
                points_collection.append(point_world.flatten().tolist())
                
        instances[class_id].append(points_collection)
    print("End")
    print("Processing video", video, "done")
    print("Time used:", round(time.time() - start, 2))
    print("============================================")


if __name__ == "__main__":
    
    input_dir = '/Users/zhiyu/Desktop/depth_instances'
    
    
    # The videos to process
    folder_names = os.listdir(input_dir)
    videos = []
    for folder in folder_names:
        if os.path.isdir(input_dir+'/' + folder):
            videos.append(folder)
    videos.sort()
    
    w = 640
    h = 480
    
    # Process for each video
    for video in videos:
        
        instances =  [[] for i in range(22)] # 21 classes, for the class id can be index directly
        generate_points(video)
    
    for i in range(len(instances)):
        if len(instances[i]) > 0:
            all_points = []
            for instance in instances[i]:
                all_points  = all_points  + instance
            t_points = np.array(all_points, dtype = np.float32)
            p = pcl.PointCloud(t_points)
            pcl.save(p, "class_" + str(i) + ".pcd", format = 'pcd')  
            
            
        
            
            
            
            
            
            