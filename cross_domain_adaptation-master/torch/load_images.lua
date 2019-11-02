require 'image'
local tdsh = require 'tds'
local rand_image_path = os.getenv('RANDIMGPATH')
assert(rand_image_path,'need to set environment variable RANDIMGPATH with the folder which contains images to crop patches')
assert(paths.dirp(rand_image_path),'Directory does not exist: '..rand_image_path)

local rand_imagelist_path = os.getenv('RANDIMGLISTPATH')

local imglist = tdsh.Vec()

print('Using random patches from '..rand_image_path)

local function lines_from(file)
  -- get all lines from a file, returns an empty 
  -- list/table if the file does not exist
  if not paths.filep(file) then return {} end
  local lines = {}
  for line in io.lines(file) do 
    table.insert(lines,line)
  end
  return lines
end

if rand_imagelist_path and paths.extname(rand_imagelist_path) == 'txt' then
  print('Using image selection from '..rand_imagelist_path)
  local imlist_temp = lines_from(rand_imagelist_path)
  for k,v in ipairs(imlist_temp) do
    imglist:insert(paths.concat(rand_image_path,v..'.jpg'))
  end
else
  for imfile in paths.iterfiles(rand_image_path) do
    imglist:insert(paths.concat(rand_image_path,imfile))
  end
end

--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------
--print 'Function Load images from mathieu chairs'

local function randomCrop(I,sampleSize)
  local iW = I:size(3)
  local iH = I:size(2)
  local sampleSize = sampleSize or {3,227,227}
  -- do random crop
  local oW = sampleSize[3]
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local out = image.crop(I, w1, h1, w1 + oW, h1 + oH)
  return out
end

local function selectRandomBG(imSize)
  local imgidx = torch.random(1,#imglist)
  local bg_mask = image.load(imglist[imgidx],3,'float')
  local msize = bg_mask:size()
  local minsize = 0
  if msize:size() == 2 then
    minsize = msize[1] < msize[2] and msize[1] or msize[2]
  else
    minsize = msize[2] < msize[3] and msize[2] or msize[3]
  end
  local dim_m = math.min(torch.random(40,120),minsize-10)
  local sampleSize = {3,dim_m,dim_m}
  bg_mask = randomCrop(bg_mask,sampleSize)
  bg_mask = image.scale(bg_mask,imSize[3],imSize[2])
  if bg_mask:dim() == 2 or bg_mask:size(1) == 1 then
    bg_mask = bg_mask:repeatTensor(3,1,1)
  end
  return bg_mask
end

local function addRandomBG(I_orig,mask_tol)
  local mask_tol = mask_tol or 254/255
  local nChns = I_orig:size(1)
  local mask = torch.sum(I_orig,1):ge(mask_tol*nChns):repeatTensor(3,1,1):byte()
  local bg_mask = selectRandomBG(I_orig:size()) -- I
  if opt and opt.color == 'gray' and bg_mask:size(1) == 3 then
    bg_mask = image.rgb2y(bg_mask)
  end

  bg_mask = image_transformer:preprocess(bg_mask):float()
  local I_rand = image_transformer:preprocess(I_orig):float()
  I_rand[mask] = bg_mask[mask]
  return I_rand
end

--------------------------------------------------------------------------------
-- RCNN cropper
--------------------------------------------------------------------------------

local function getRCNNCrop(res,I,bbox)

  local crop_size = crop_size or 227
  local padding = 16
  local use_square = false

  local pad_w = 0;
  local pad_h = 0;
  local crop_width = crop_size;
  local crop_height = crop_size;

  ------
  if padding > 0 or use_square then
    local scale = crop_size/(crop_size - padding*2)
    local half_height = (bbox[4]-bbox[2]+1)/2
    local half_width = (bbox[3]-bbox[1]+1)/2
    local center = {bbox[1]+half_width, bbox[2]+half_height}
    if use_square then
      -- make the box a tight square
      if half_height > half_width then
        half_width = half_height;
      else
        half_height = half_width;
      end
    end
    bbox[1] = torch.round(center[1] - half_width  * scale)
    bbox[2] = torch.round(center[2] - half_height * scale)
    bbox[3] = torch.round(center[1] + half_width  * scale)
    bbox[4] = torch.round(center[2] + half_height * scale)

    local unclipped_height = bbox[4]-bbox[2]+1;
    local unclipped_width = bbox[3]-bbox[1]+1;
    
    local pad_x1 = math.max(0, 1 - bbox[1]);
    local pad_y1 = math.max(0, 1 - bbox[2]);
    -- clipped bbox
    bbox[1] = math.max(1, bbox[1]);
    bbox[2] = math.max(1, bbox[2]);
    bbox[3] = math.min(I:size(3), bbox[3]);
    bbox[4] = math.min(I:size(2), bbox[4]);
    local clipped_height = bbox[4]-bbox[2]+1;
    local clipped_width = bbox[3]-bbox[1]+1;
    local scale_x = crop_size/unclipped_width;
    local scale_y = crop_size/unclipped_height;
    crop_width = torch.round(clipped_width*scale_x);
    crop_height = torch.round(clipped_height*scale_y);
    pad_x1 = torch.round(pad_x1*scale_x);
    pad_y1 = torch.round(pad_y1*scale_y);

    pad_h = pad_y1;
    pad_w = pad_x1;

    if pad_y1 + crop_height > crop_size then
      crop_height = crop_size - pad_y1;
    end
    if pad_x1 + crop_width > crop_size then
      crop_width = crop_size - pad_x1;
    end
  end -- padding > 0 || square
  ------

  local patch = I[{{},{bbox[2],bbox[4]},{bbox[1],bbox[3]}}];
  local tmp = image.scale(patch,crop_width,crop_height,'bilinear');

  res:resize(3,crop_size,crop_size):zero()

  res[{{},{pad_h+1,pad_h+crop_height}, {pad_w+1,pad_w+crop_width}}] = tmp

end

local function PatchProvider(batch_size,nThreads,iter_per_thread)
  local imlist_real = imglist
  local bbox_path = assert(os.getenv('BBOXFILEPATH'),'need to provide bounding box file name')--'/home/francisco/work/libraries/rcnn/data/selective_search_data/voc_2007_train.mat'
  local matioh = require 'matio'
  matioh.use_lua_strings = true
  print('Verifying integrity of imagelist and bboxes')
  local dt = matioh.load(bbox_path)
  for i=1,#imlist_real do
    assert(paths.basename(imlist_real[i],'jpg')==dt.images[i], 'Bboxes and image names does not match!')
  end
  dt = nil
  collectgarbage()
  
  local crop_size = crop_size or 227
  local iter_per_thread = iter_per_thread or 8
  local nThreads = nThreads or 6
  local image_transf = image_transformer
  local Threads = require 'threads'
  Threads.serialization('threads.sharedserialize')
  local donkeys = Threads(nThreads,
              function()
                require 'torch'
                require 'image'
                require 'paths'
                tds = require 'tds'
                require 'nnf'
                matio = require 'matio'
                matio.use_lua_strings = true
              end,
              function(idx)
                crop_size = crop_size
                torch.setheaptracking(true)
                torch.manualSeed(idx)
                imagelist_real = imlist_real
                nImages_real = #imagelist_real
                local dt = matio.load(bbox_path)
                collectgarbage()
                roidb = tds.Vec()
                -- compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
                for i=1,#dt.images do
                  roidb:insert(dt.boxes[i]:index(2,torch.LongTensor{2,1,4,3}):int())
                end
                dt = nil
                collectgarbage()
                getCrop = getRCNNCrop
                image_transformer = image_transf
                img_buffer = torch.FloatTensor(iter_per_thread,3,crop_size,crop_size)
                iter_per_thread = iter_per_thread--]]
                print(string.format('Starting donkey with id: %d', idx))
             end
              )

  local images = torch.FloatTensor(batch_size,3,crop_size,crop_size)
  
  local function retfunc()
    for i=1,batch_size/iter_per_thread do
      donkeys:addjob(
        function()-- [[
          -- load images
          for j=1,iter_per_thread do
            local imidx = torch.random(1,nImages_real)
            local I = image.load(imagelist_real[imidx],3,'float')
            I = image_transformer:preprocess(I)
            local bbidx = torch.random(1,roidb[imidx]:size(1))
            getCrop(img_buffer[j],I,roidb[imidx][bbidx])
          end
          collectgarbage()--]]
          return img_buffer
        end,
        function(I)
          images:narrow(1,(i-1)*iter_per_thread+1,iter_per_thread):copy(I)
          --print(I)
        end
      )
    end
    donkeys:synchronize()
    return images
  end
  
  return retfunc
end

return addRandomBG, getRCNNCrop, PatchProvider
