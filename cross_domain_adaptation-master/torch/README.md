## How to use this code

### Dependencies
Installation instructions present in each link

  * [loadcaffe](https://github.com/szagoruyko/loadcaffe)
  * [hdf5](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)
  * [matio](https://github.com/soumith/matio-ffi.torch)
  * **tds** run `luarocks install tds`

Optional (but good to have, maybe already installed by default)
  * [cudnn](https://github.com/soumith/cudnn.torch)

### Running the code
Have a look at `example_run.sh`. It sets the required environment variables needed for running the code.



#### Quick explanation
This code supposes that the CAD models have already been cropped with a small padding around it (15% of the tight crop). The initial images can have any size, the scaling is done inside the function. For an idea on how I cropped the images, have a look at `matlab/mathieu_renders/create_imagelist_mathieu_renders.m`

**Important**: The CAD folder structure is supposed to be respect the following pattern: each cad render is in a separate folder in the rootfolder, with each cad model having the same number of rendered images !
```
rootfolder/
-----------cad_model1/
----------------------renders from cad_model1
-----------cad_model2/
----------------------renders from cad_model2
```

##### Generic functions
  * `save_calibration_features.lua` loads the CAD images, adds 1 random background per CAD view, and save both raw and backgrounded features to disk.
  * `learn_projection.lua` reads the saved features from disk, and trains a projection with them.

##### Nearest neighbor detection
  * `calibrate_scores.lua` reads the CAD features, computes features for random image patches and calibrate them.
  * `detect.lua` do the detection in the nearest neighbor approach

##### Logistic regression
  * `baseline_logistic_train.lua`
  * `baseline_logistic_detect.lua`


#### A few more explanations
##### Visualizing the CAD with random background
If you want to visualize the random backgrounds added on top of the cad models, run `RANDIMGPATH=/path/to/random/images qlua` and execute the following code (don't forget to load a CAD model called `I` before !):
```lua
require 'torch'
require 'image'
require 'os'
require 'nnf'
-- applies RGB->BGR, scaling and mean subtraction.
-- default arguments doesn't do anything
image_transformer = nnf.ImageTransformer{}
addRandomBG = dofile 'load_images.lua'

-- select a CAD render you want, lets call it I
mask_tol = 240/255 -- every pixels whose 3 channels are >= that is background
I_background = addRandomBG(I, mask_tol)
image.display(I)
image.display(I_background)
```

###### Detection
For the detection, you need to open `load_dataset.lua` and change the variables
`im_dir` (folder which contains the images) and `ss_path` (path to the .mat file
with the selective search bounding boxes) to their corresponding paths in your system.

##### Command Line check in torch
Just to say that when a variable is boolean in the command line default parameter,
then setting it (for example `-relu`) inverts the default value. Don't use it as
`-relu true` or `-relu false`.

#### Selective Search bounding boxes
The pre-computed bounding boxes from VOC2007 and VOC2012 can be found at
```
http://www.cs.berkeley.edu/~rbg/r-cnn-release1-selective-search.tgz
```

