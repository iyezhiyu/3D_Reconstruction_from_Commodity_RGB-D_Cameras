# 3D Reconstruction from Commodity RGB-D Cameras
## Contents
* [Introduction](#Introduction)
* [Modules](#Modules)
* [Frame Extraction](#Frame_Extraction)
* [Mask R-CNN](#Mask_R-CNN)
* [Point Cloud](#Point_Cloud)
* [TSDF Volume](#TSDF_Volume)
* [Inpainting and Refinement](#Inpainting_and_Refinement)


## Introduction
This project aims to reconstruct the 3D shapes of the objects from the output of the commodity RGB-D cameras of home robots. There are eight modules in the pipeline, however, some modules are implemented together.
## Frame_Extraction
* average_extractor.py: extract frames of testing videos (video 0048 - 0059) of the YCB Video Dataset; results are in the file extract_result_2
## Mask_R-CNN
The second module is Mask R-CNN. We use [Detectron](https://github.com/facebookresearch/Detectron) from Facebook AI Research.
### Installation
* Install the requirements for running Detectron, refer to [INSTALL.md](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)
* Install [OpenCV](https://opencv.org) and [Shapely](https://pypi.org/project/Shapely/) for annotation generation
### Annotation generation for training
* video_data_annotations_generator.py: generate the annotations for the training videos (0000 - 0047, 0060 - 0091) of the YCB Video Dataset as the format of the [COCO Dataset](http://cocodataset.org/#home), the tools/sub_masks_annotations.py is demanded for running this program
### Files in the Detectron which need to modify
#### $Detectron/detectron/datasets/dataset_catalog.py
* In the _DATASETS dictionary, add
```
    'ycb_video': {
            _IM_DIR:
            'path to/YCB_Video_Dataset/data',
            _ANN_FN:
            'path to/YCB_Video_Dataset/annotations/instances.json'
    },
```
#### $Detectron/detectron/datasets/dummy_datasets.py
* Modify classes as \_\_background\_\_ and the classes we use
```
    classes = [
        '__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box',
        '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box',
        '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base',
        '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', 
        '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
    ]
```
#### Edit the configuration file
* Choose one of the configuration file of the Mask RCNN in $Detectron/configs, such as $Detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml
```
    MODEL:
      NUM_CLASSES: 22  # (1 (background) + number of classes)
    TRAIN:
      DATASETS: ('ycb_video',)
      SCALES: (480,)
      MAX_SIZE: 640 
```
* Also comment all the codes in the TEST of the configuration file, because we do the inference using infer_simple.py
* Other parameters such as NUM_GPUS, BASE_LR, MAX_ITER, etc., can be modified if needed.
* It is recommended that the BASE_LR should be set to a smaller value, such as 0.001, in order to addressing the "Loss is NaN" error.
#### $Detectron/tools/infer_simple.py
##### In order to output a json file contains the segmentations results, some codes should be added.
* In the beginning of the program
```
    import json
    import numpy as np
    import pycocotools.mask as mask_utils
```
* Before the for-loop of images
```
    json_output = []
```
* In the for-loop of images
```
    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format( cls_boxes, cls_segms, cls_keyps)

    if boxes is None:
        boxes = []
    else:
        boxes = boxes.tolist()
        
    segmentations = mask_utils.decode(segms)
    if segmentations is None:
        segmentations = []
    else:
        segmentations = np.swapaxes(segmentations, 0, 2)
        segmentations = np.swapaxes(segmentations, 1, 2)
        segmentations = segmentations.tolist()

    json_output.append({ \
        'image_name': im_name,
        'boxes': boxes,
        'classes': classes,
        'segms': segmentations
    })
```
* After the for-loop of images
```
    with open(args.output_dir + '/annotations.json', 'w') as outfile:
        json.dump(json_output, outfile)
```
#### $Detectron/detectron/utils/env.py
* In order to addressing the error of "yaml.constructor.ConstructorError", replace
```
    yaml_load = yaml.load
```
* with
```
    yaml_load = lambda x: yaml.load(x, Loader = yaml.Loader)
```
### Training step
* In the $Detectron directory
```
    python tools/train_net.py \
      --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
      OUTPUT_DIR out_dir
```
* This step generates a trained Mask R-CNN model (model_fianl.pkl) for detection.
### Detection
* Step a in the pipeline: detect the images of the test videos (video 0048 - 0059) of the YCB Video Dataset
```
    python tools/infer_simple.py \
      --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
      --output-dir path to extract_result_annotations/name of video \
      --image-ext png \
      --wts path to/model_final.pkl \
      path to extract_result_2/name of video/color
```
## Depth_Information_for_Each_Object
* Step c in the pipeline: using depth_instances_extractor.py to extract the depth information for each instance; results are in the file depth_instances
## Point_Cloud
* Generate the point cloud of each object in each video for visualisation using point_cloud_generator.py
## TSDF_Volume
### Installation
#### Install [PyFusion](https://github.com/griegler/pyfusion)
* Step 1: make sure the gcc version is no more than 7.
* Step 2: follow the instructions in the README of the PyFusion.
* Step 3: set path as follows.
```
    export LD_LIBRARY_PATH=path_to/pyfusion/build:$LD_LIBRARY_PATH
    export PYTHONPATH=path_to/pyfusion:$PYTHONPATH
    export PYTHONPATH=path_to/:$PYTHONPATH
```
### Generation
* Step d in the pipeline: using tsdf_generator.py to generate the TSDF volume for each object in each video
## Inpainting_and_Refinement
### Installation
* refer to [GenRe-ShapeHD](https://github.com/xiumingzhang/GenRe-ShapeHD)
### Inpainting and refinement
* move inpaint_refine.py and inpaint_refine.sh to $GenRe-ShapeHD
* run inpaint_refine.sh
* this includes step e,f,g,h in the pipeline, thie process outputs all we need for this project, including spherical maps and final 3D shapes