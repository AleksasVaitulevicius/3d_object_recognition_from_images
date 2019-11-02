require 'image'
require 'paths'
local matio = require 'matio'

local im_dir = os.getenv('IMGPATHTEST')
assert(im_dir,'need to set environment variable IMGPATHTEST with the folder which will contain all the test images for detection')

local ss_path = os.getenv('BBOXFILEPATHTEST')
assert(ss_path,'need to set environment variable BBOXFILEPATHTEST with the file that contains the bounding boxes for detection')

assert(paths.dirp(im_dir),'Image path for detection does not exist: '..im_dir)
assert(paths.filep(ss_path),'Bounding box file for detection does not exist: '..ss_path)

local img_ext = '.jpg'

local ds = {}
do
  ds.im_dir = im_dir
  local dt = matio.load(ss_path)
  ds.im_idx = dt.images
  -- get Image
  ds.getImage = function(self,i)
    return image.load(paths.concat(self.im_dir,self.im_idx[i]..img_ext),3,'float')
  end
  -- dataset size
  function ds:size()
    return #self.im_idx
  end
  function ds:getName(i) return paths.basename(self.im_idx[i],'jpg') end
  -- bboxes
  function ds:loadROIDB()
    --local dt = matio.load(ss_path)
    local img2roidb = {}
    -- compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
    for i=1,#dt.images do
      img2roidb[dt.images[i] ] = i
    end
    ds.roidb = {}
    for i=1,ds:size() do
      table.insert(self.roidb, dt.boxes[img2roidb[self.im_idx[i] ] ]:index(2,torch.LongTensor{2,1,4,3}):int())
    end
  end
  ds:loadROIDB()
end

return ds
