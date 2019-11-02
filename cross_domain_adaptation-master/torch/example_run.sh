#!/bin/bash

# set GPU to use
GPU=1 # 0-based

################################################################################
## Data parameters
################################################################################

# Cache folder where all experiments will go to
export CACHEFOLDERPATH="/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/projects/cross_domain/cachedir/"

## set it once and it should be fine for the training

export RANDIMGPATH="/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit/VOC2007/JPEGImages"
export RANDIMGLISTPATH="/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
export BBOXFILEPATH="/home/francisco/work/libraries/rcnn/data/selective_search_data/voc_2007_train.mat"

##

## Pretrained Model in Caffe format
export CAFFENETPROTOTXT="/home/francisco/work/libraries/caffe/examples/imagenet/imagenet_deploy.prototxt"
export CAFFENETCAFFEMODEL="/home/francisco/work/libraries/caffe/examples/imagenet/caffe_reference_imagenet_model"
export USECUDNN=1 


################################################################################
## Saving CAD features
################################################################################
# This is the path to the folder containing the CAD renders, as explained
# in the README.
export CADIMGPATH="/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/crops/sphere_renders_perspective_object_new"

# Name this CAD renders as you want, but be consistent because the other functions
# will look for this NAME to load the features
NAME="ikeaobject"

LAYER="conv4" # currently, we have conv3, conv4, conv5, fc6 and fc7
COLOR="gray" # rgb or gray

# Computes the features for the CAD models in CADIMGPATH, naming it NAME, using
# random patches from RANDIMGPATH
CUDA_VISIBLE_DEVICES=$GPU th save_calibration_features.lua -name $NAME -color $COLOR -layer $LAYER

# From now on, we will read the features from the CAD models with NAME

################################################################################
## Learn a projection
################################################################################

# There are lots of parameters here, type th learn_projection.lua --help for
# an overview

# for more complex projection types, define -conv_proj. Its possible values are
# "1x1", "3x3", "2t1x1", "2t3x3", "1x1relu" and "3x3relu"

CUDA_VISIBLE_DEVICES=$GPU th learn_projection.lua -name $NAME -color $COLOR -layer $LAYER


################################################################################
## Calibrate
################################################################################

# once again, lots of parameters, type th calibrate_scores.lua --help to see the
# possible combinations

# note that it will read the features and the calibration that were previously
# computed, so make sure that you have the good set of parameters

NTHREADS=4 # for parallel data loading, if it crashes, reduce this number

# NAME is the CAD features you are calibrating
NAME="ikeaobject"
# PNAME specifies the dataset of the projection you will use. Can be the same
# as NAME. As we have in this example computed the projections in ikeaobject
# I'll use the same here
PNAME="ikeaobject"


CUDA_VISIBLE_DEVICES=$GPU th calibrate_scores.lua -color $COLOR -layer $LAYER -name $NAME -projname $PNAME


################################################################################
## Detection
################################################################################

# Need to set some more environment variables here
# They correspond to the different datasets for detection that we used
# I'll add the three here just to have them, uncomment the ones you want

# for IKEAobject detection
export IMGPATHTEST="/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/3d_toolbox_notfinal/data/img/"
export BBOXFILEPATHTEST="/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/ikea_object/ss_boxes.mat"
DATASETNAME="ikeaobject"

# for VOC2007
#export IMGPATHTEST="/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit/VOC2007/JPEGImages"
#export BBOXFILEPATHTEST="/home/francisco/work/libraries/rcnn/data/selective_search_data/voc_2007_test.mat"
#DATASETNAME="voc2007"

# for VOC2012 easy subset
#export IMGPATHTEST="/home/francisco/work/datasets/VOCdevkit/VOC2012/JPEGImages"
#export BBOXFILEPATHTEST="/home/francisco/work/projects/cross_domain/torch_new/mathieu_chairs/mathieu_chairs_ss_bboxes.mat"
#DATASETNAME="voc2012subset"

CUDA_VISIBLE_DEVICES=$GPU th detect.lua -color $COLOR -layer $LAYER -name $NAME -projname $PNAME -dataset $DATASETNAME

