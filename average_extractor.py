#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:18:07 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""

import os
import shutil


if __name__ == "__main__":
    #path of the YCB Video Dataset
    videopath = '/Users/zhiyu/Desktop/YCB_Video_Dataset/data'
    videos = ["0048", "0049", "0050", "0051", "0052", "0053", \
              "0054", "0055", "0056", "0057", "0058", "0059"]
    #Directory to store the processed frames
    out_dir = '/Users/zhiyu/Desktop/extract_result_2'
    #Frames per second
    frames_per_second = 30
    #Frames needed per second
    num_of_frames_per_second = 1
    #interval of frames extracted
    interval = round(frames_per_second / num_of_frames_per_second)
    
    for video in videos:
        if os.path.isdir(videopath + '/' + video):
            target_video = videopath + '/' + video
            extract_outdir = out_dir + '/' + video
            if not os.path.exists(extract_outdir):
                    os.makedirs(extract_outdir)
            print("============================================")
            print("Target video :", target_video)
            print("Frame save directory:", extract_outdir)
            files = os.listdir(target_video)
            color_files = [file[:6] for file in files if file[7:12] == 'color']
            color_files.sort()
            if not os.path.exists(extract_outdir + '/color'):
                os.makedirs(extract_outdir + '/color')
            for image_id in color_files:
                if (int(image_id) - 1) % interval == 0:
                    shutil.copy(target_video +  '/'  + image_id + '-color.png', extract_outdir + '/color/' + image_id + '-color.png')
                    shutil.copy(target_video +  '/'  + image_id + '-depth.png', extract_outdir + '/' + image_id + '-depth.png')
                    shutil.copy(target_video +  '/'  + image_id + '-meta.mat', extract_outdir + '/' + image_id + '-meta.mat')
            print("============================================")
            
    print(" (●ﾟωﾟ●) Extraction Done! (づ｡◕‿‿◕｡)づ")