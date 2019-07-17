#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:15:34 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""
"""
For a home robot, the threshold method is more suitable.
The threshold is low because the images from a video are continuous.
"""

import os
import shutil
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema

 
def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
 
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   #print(x)
   return x
 
    
if __name__ == "__main__":
    #print(sys.executable)
    #Using the threshold method because it is suitable for a home robot
    #Setting fixed threshold criteria
    USE_THRESH = True
    #fixed threshold value
    THRESH = 0.28
    #Setting fixed threshold criteria
    USE_TOP_ORDER = False
    #Setting local maxima criteria
    USE_LOCAL_MAXIMA = False
    #Number of top sorted frames
    NUM_TOP_FRAMES = 50
     
    #path of the YCB Video Dataset
    videopath = '/Users/zhiyu/Desktop/YCB_Video_Dataset/data'
    videos = os.listdir(videopath)
    #Directory to store the processed frames
    out_dir = '/Users/zhiyu/Desktop/extract_result'
    #smoothing window size
    len_window = int(50)
    
    for video in videos:
        if os.path.isdir(videopath + '/' + video):
            target_video = videopath + '/' + video
            extract_outdir = out_dir + '/' + video
            print("============================================")
            print("Target video :", target_video)
            print("Frame save directory:", extract_outdir)
            # load video and compute diff between frames
            files = os.listdir(target_video)
            color_files = [file for file in files if file[7:12] == 'color']
            color_files.sort()
            
            curr_frame = None
            prev_frame = None 
            frame_diffs = []
            frames = []
            i = 0
            while(i < len(color_files)):
                frame = cv2.imread(target_video + '/' + color_files[i])
                luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
                curr_frame = luv
                if curr_frame is not None and prev_frame is not None:
                    #logic here
                    diff = cv2.absdiff(curr_frame, prev_frame)
                    diff_sum = np.sum(diff)
                    diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
                    frame_diffs.append(diff_sum_mean)
                    frame = Frame(i, diff_sum_mean)
                    frames.append(frame)
                prev_frame = curr_frame
                i = i + 1
                
            
            # compute keyframe
            keyframe_id_set = set()
            if USE_TOP_ORDER:
                # sort the list in descending order
                frames.sort(key=operator.attrgetter("diff"), reverse=True)
                for keyframe in frames[:NUM_TOP_FRAMES]:
                    keyframe_id_set.add(keyframe.id) 
            if USE_THRESH:
                print("Using Threshold")
                for i in range(1, len(frames)):
                    if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                        keyframe_id_set.add(frames[i].id)   
            if USE_LOCAL_MAXIMA:
                print("Using Local Maxima")
                diff_array = np.array(frame_diffs)
                sm_diff_array = smooth(diff_array, len_window)
                frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
                for i in frame_indexes:
                    keyframe_id_set.add(frames[i - 1].id)
                    
                plt.figure(figsize=(40, 20))
                plt.locator_params(numticks=100)
                plt.stem(sm_diff_array)
                plt.savefig(out_dir + 'plot.png')
            
            
            print('Average frames per second:', round(len(keyframe_id_set)/(len(color_files)/30), 2))
            # save all keyframes as image
            
            if not os.path.exists(extract_outdir):
                    os.makedirs(extract_outdir)
            idx = 0
            while(idx < len(color_files)):
                frame = cv2.imread(target_video + '/' + color_files[idx])
                if idx in keyframe_id_set:
                    cv2.imwrite(extract_outdir + '/' + color_files[idx], frame)
                    shutil.copy(target_video +  '/'  + color_files[idx][:-9] + 'depth.png', extract_outdir + '/' +color_files[idx][:-9] + 'depth.png')
                    shutil.copy(target_video +  '/'  + color_files[idx][:-9] + 'meta.mat', extract_outdir + '/' +color_files[idx][:-9] + 'meta.mat')
                    keyframe_id_set.remove(idx)
                idx = idx + 1
            
            print("============================================")
    
    print(" (●ﾟωﾟ●) Extraction Done! (づ｡◕‿‿◕｡)づ")