#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:59:16 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
This program aims to generate TSDF of each object.
"""
import os
import time
import json
import imageio
import pyfusion
import numpy as np
from collections import Counter


if __name__ == "__main__":
    
    input_dir = '/mnt/disk/zhiyu/depth_instances'
    output_dir = '/mnt/disk/zhiyu/video_objects'

    
    # The videos to process
    folder_names = os.listdir(input_dir)
    videos = []
    for folder in folder_names:
        if os.path.isdir(input_dir+'/' + folder):
            videos.append(folder)
    videos.sort()
    
    
    # Process for each video
    for video in videos:
        
        start = time.time()
        print("============================================")
        print("Processing video :", video)
        print("Begin")

        json_output = []
        
        video_dir = input_dir + '/' + video
        with open(video_dir + '/annotations.json','r') as load_f:
            annotations = json.load(load_f)
        
        def take_class(elem):
            return [elem['class'],elem['image_name']]
        annotations.sort(key = take_class)
        
        the_class = []
        for annotation in annotations:
            the_class.append(annotation["class"])
        num_class = Counter(the_class)
        the_class = list(set(the_class))
        
        start_index, end_index = 0, 0
        for this_class in the_class:
            print("Processing class", this_class)
            depth_images = []
            Ks = []
            Rs = []
            Ts = []
            print(num_class[this_class])
            start_index = end_index
            end_index += num_class[this_class]

# =============================================================================
#             volume = open3d.integration.ScalableTSDFVolume(voxel_length = 1.0/128.0, sdf_trunc = 0.05, color_type = open3d.integration.TSDFVolumeColorType.RGB8)
#         
#             fake_color_image = np.array([[[0.5,0.5,0.5]for u in range(640)] for v in range(480)]).astype('float32')
#             for i in range(start_index, end_index):
#                 image_name = annotations[i]['image_name']
#                 depth_image = open3d.io.read_image(video_dir + '/' + image_name)
#                 #rotation_translation_matrix = np.linalg.inv(np.vstack((np.array(annotation['rotation_translation_matrix']), np.array([0,0,0,1]))))
#                 rotation_translation_matrix = np.vstack((np.array(annotation['rotation_translation_matrix']), np.array([0,0,0,1])))
#                 factor_depth = annotations[i]["factor_depth"]
#                 K = np.array(annotations[i]['intrinsic_matrix'])
#                 color_image = open3d.geometry.Image(fake_color_image)
#                 rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(color_image, depth_image, depth_scale = factor_depth, depth_trunc = 4.0, convert_rgb_to_intensity=False)
#                 intrinsic = open3d.camera.PinholeCameraIntrinsic(w, h, K[0][0], K[1][1], K[0][2], K[1][2])
#                 volume.integrate(rgbd, intrinsic, rotation_translation_matrix)
#                 
#         mesh = volume.extract_triangle_mesh()
#         mesh.compute_vertex_normals()
#         open3d.visualization.draw_geometries([mesh])
# =============================================================================

            for i in range(start_index, end_index):
                
                image_name = annotations[i]['image_name']
                #print("Processing", image_name)
                
                factor_depth = annotations[i]["factor_depth"]
                depth_image = np.array(imageio.imread(video_dir + '/' + image_name)/factor_depth).astype('float32')
                depth_images.append(depth_image)
                
                K = np.array(annotations[i]['intrinsic_matrix']).astype('float32')
                Ks.append(K)
                
                #rotation_translation_matrix = np.linalg.inv(np.vstack((np.array(annotation['rotation_translation_matrix']), np.array([0,0,0,1]))))
                rotation_translation_matrix = np.array(annotation['rotation_translation_matrix'])
                R = np.array(rotation_translation_matrix[0:3,0:3]).astype('float32')
                T = np.array(rotation_translation_matrix[0:3,3]).astype('float32')
                Rs.append(R)
                Ts.append(T)
                
            depth_images = np.array(depth_images)
            Ks = np.array(Ks)
            Rs = np.array(Rs)
            Ts = np.array(Ts)

            truncate = 0.025
            views = pyfusion.PyViews(depth_images, Ks, Rs, Ts)
            tsdf = pyfusion.tsdf_gpu(views, 128, 128, 128, 1.0/128, truncate, False)
            #xx = []
            #yy = []
            #zz = []
            tsdf_reshape = np.reshape(tsdf, (128 * 128 * 128))
            tsdf_num = set(list(tsdf_reshape))
            #print(tsdf_num)
            num1 = sum(np.where(tsdf_reshape == truncate, 1, 0))
            num2 = sum(np.where(tsdf_reshape == -truncate, 1, 0))
            print(num1)
            print(num2)
            print(128 * 128 * 128 - num1 - num2)
            #for mi in range(128):
            #    for mj in range(128):
            #        for mk in range(128):
            #            if abs(tsdf[0][mi][mj][mk]-truncate) < 0.0000001 and abs(tsdf[0][mi][mj][mk]+truncate) < 0.0000001:
            #                xx.append(mi)
            #                yy.append(mj)
            #                zz.append(mk)
            #print(len(xx))
            json_output.append({'class':this_class,'tsdf_list':tsdf_reshape.tolist()})
        
        video_out_dir = output_dir + '/' + video
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        with open(video_out_dir + '/tsdf.json', 'w') as f:
            json.dump(json_output, f)
        
        print("End")
        print("Processing video", video, "done")
        print("Time used:", round(time.time() - start, 2))
        print("============================================")
        
