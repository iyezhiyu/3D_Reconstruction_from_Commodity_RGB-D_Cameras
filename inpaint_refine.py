#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:00:39 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
This program aims to project tsdf to the spherical map, and use the pretrained
spherical map inpainting network provided by GenRe to get an inpainted spherical
map.
Next, use the original voxel (tsdf) and the voxel projected from the inpainted
spherical map to generate the final 3D shape.
"""
import os
import json
import time
import numpy as np
from shutil import rmtree
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.spherical_proj import render_spherical, sph_pad
from options import options_test
import datasets
import models
import util.util_loadlib as loadlib
from util.util_print import str_error, str_stage, str_verbose
from loggers import loggers
from visualize.visualizer import Visualizer


def generate_categories():
    class_file = open(YCB_Video_Dataset_path + '/image_sets/classes.txt')
    line = class_file.readline()
    category_id = 0
    category_name2id = {}
    category_id2name = {}
    while line:
        category_id += 1
        category_name2id[line[:-1]] = category_id
        category_id2name[str(category_id)] = line[:-1]
        line = class_file.readline()
    class_file.close()
    return category_name2id, category_id2name


def isborder(i,j,k):
    if i == 0 or i == 127 or j == 0 or j == 127 or k == 0 or k == 127:
        return True
    return False


def tsdf_postprocess(tsdf):
    new_tsdf = np.array([[[0.0 for i in range(128)] for j in range(128)] for k in range(128)])
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if isborder(i,j,k):
                    new_tsdf[i][j][k] = tsdf[i][j][k]
                else:
                    if tsdf[i-1][j][k] != 0 and tsdf[i+1][j][k] != 0 and tsdf[i][j-1][k] != 0 and tsdf[i][j+1][k] != 0 and tsdf[i][j][k-1] != 0 and tsdf[i][j][k+1] != 0:
                        new_tsdf[i][j][k] = 0
                    else:
                        new_tsdf[i][j][k] = tsdf[i][j][k]
    return new_tsdf


if __name__ == "__main__":
   

    # the options
    opt = options_test.parse()
    opt.full_logdir = None
    # many arguments are nonsense, just print for checking the program can run
    # print(opt)


    # set the gpu
    if opt.gpu == '-1':
        device = torch.device('cpu')
    else:
        loadlib.set_gpu(opt.gpu)
        device = torch.device('cuda')


    # in my program, the manual seed is None
    if opt.manual_seed is not None:
        loadlib.set_manual_seed(opt.manual_seed)


    # set the paths
    YCB_Video_Dataset_path = '/mnt/disk/zhiyu/YCB_Video_Dataset'
    input_dir = '/mnt/disk/zhiyu/video_objects'
    output_dir = opt.output_dir
    #output_dir += ('_' + opt.suffix.format(**vars(opt))) \
    #        if opt.suffix != '' else ''
    opt.output_dir = output_dir
    if os.path.isdir(output_dir):
        if opt.overwrite:
            rmtree(output_dir)
        else:
            raise ValueError(" %s already exists, but no overwrite flag" % output_dir)
    os.makedirs(output_dir)

    
    # class id to name conversion
    _, category_id2name = generate_categories()
    
    
    # set the loggers just for use the pretrained full model
    logger_list = [
            loggers.TerminateOnNaN(),
    ]
    logger = loggers.ComposeLogger(logger_list)
    

    # The videos to process
    folder_names = os.listdir(input_dir)
    videos = []
    for folder in folder_names:
        if os.path.isdir(input_dir+'/' + folder):
            videos.append(folder)
    videos.sort()
    
    
    # set the visualizer
    my_visualizer = Visualizer(
            n_workers = getattr(opt, 'vis_workers', 4),
            param_f = getattr(opt, 'vis_param_f', None)
    )
    

    # set the models to use
    # first, set the Spherical map rendering
    render_spherical_map = render_spherical().to(device)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('The model for rendering spherical maps has been set successfully')
    # second, set the genre full model. In this program, use the second model which
    # is spherical map inpainting net, to inpaint the spherical map, and use the refine
    # model which refines the voxels to get the final shape
    Full_model = models.get_model(opt.net, test=True)
    full_model = Full_model(opt, logger)
    full_model.to(device)
    full_model.eval()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('The model for inpainting sphercial maps has been set successfully')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('The model for refining voxels has been set successfully')

    
    # Process for each video
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print("Process videos")
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for video in videos:
        

        start = time.time()
        print("============================================")
        print("Processing video :", video)
        print("Begin")


        json_output = []
        

        with open(input_dir + '/' + video + '/tsdf.json','r') as load_f:
            tsdf_lists = json.load(load_f)
        
        video_out_dir = output_dir + '/' + video
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        
        for p in range(len(tsdf_lists)):
        #for p in range(4,5):
            
            
            # object class
            this_class = tsdf_lists[p]['class']
            print("Processing class", category_id2name[str(this_class)])
            
            # the tsdf fused from depth maps
            the_object = tsdf_lists[p]['tsdf_list']
            # change the domain of -0.025~0.025 to 0~1
            tsdf = np.reshape(the_object, (128, 128, 128)) * 20 + 0.5
            # use numpy.clip instead of torch.clamp because torch.clamp has
            # a bug in the versions of 0.4.0 and 0.4.1
            clipped_tsdf = tsdf_postprocess(np.clip(tsdf, 0, 1-1e-5))
            #print(np.max(clipped_tsdf))
            #print(np.min(clipped_tsdf))
            tensor_tsdf = torch.from_numpy(np.reshape(clipped_tsdf, (1,1,128,128,128))).float().to(device)
            

            # render the tsdf to partial spherical map
            sph_in = render_spherical_map(tensor_tsdf)
            padding_margin = 16
            sph_in = sph_pad(sph_in, padding_margin)


            # inpaint the partial spherical map
            with torch.no_grad():
                out2 = full_model.net.depth_and_inpaint.net2(sph_in)


            # the voxel projected from inpainted spherical maps
            pred_proj_sph = full_model.net.backproject_spherical(out2['spherical'])
            pred_proj_sph = torch.transpose(pred_proj_sph, 3, 4)
            pred_proj_sph = torch.flip(pred_proj_sph, [3])


            # the voxel projected from depth maps
            proj = torch.transpose(tensor_tsdf, 3, 4)
            proj = torch.flip(proj, [3])
            #print(proj.size())
            
            
            # combine two voxels which need to be passed to the final refinement net
            refine_input = torch.cat((pred_proj_sph, proj), dim = 1)
            with torch.no_grad():
                pred_voxel = full_model.net.refine_net(refine_input)
            
            
            # write the intermediate results and final results to json file
            json_output.append({'class': category_id2name[str(this_class)], \
                    'partial_spherical_map': sph_in.tolist(), \
                    'inpainted_spherical_map': out2['spherical'].tolist(), \
                    'projected_voxel_from_sph': pred_proj_sph.flip([3]).transpose(3,4).tolist(), \
                    'projected_voxel_from_depth': proj.flip([3]).transpose(3,4).tolist(), \
                    'predicted_voxel': pred_voxel.flip([3]).transpose(3,4).tolist()})
            
    
            # visualization
            vis = {}
            vis['pred_voxel'] = pred_voxel.cpu().numpy()
            object_out_dir = video_out_dir + '/' + category_id2name[str(this_class)]
            if not os.path.exists(object_out_dir):
                os.makedirs(object_out_dir)
            my_visualizer.visualize(vis, 0, object_out_dir)
            
        
        # write json file for each video
        print("Writing JSON file for " + video)
        with open(video_out_dir + '/genre.json', 'w') as f:
            json.dump(json_output, f)


        print("End")
        print("Processing video", video, "done")
        print("Time used:", round(time.time() - start, 2))
        print("============================================") 
    

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('All videos processed')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')