require 'cunn'
require 'cudnn'
require 'nnf'
require 'trepl'
require 'optim'
matio = require 'matio'
require 'loadcaffe'
require 'hdf5'
paths.dofile('../Optim.lua')
tds = require 'tds'

matio.use_lua_strings = true

local rootfolder = os.getenv('CACHEFOLDERPATH')
assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

local expfolder = 'baseline_classifier'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_chairs','')
cmd:option('-projname','','')
cmd:option('-color','rgb','')--rgb
cmd:option('-model','caffenet','')
cmd:option('-layer','conv5','')
cmd:option('-normalize',false,'')
cmd:option('-linproj',true,'')
cmd:option('-conv_proj','','')
cmd:option('-rbatch',96,'n_real_feats_per_batch')
cmd:option('-sbatch',32,'n_synth_feats_per_batch')
cmd:option('-synth_with_bg',false,'') -- fixed_bg
cmd:option('-lr',1e0,'lr proj')
cmd:option('-wd',5e-4,'wd proj')
cmd:option('-lr2',0.001,'lr classif')
cmd:option('-wd2',5e-4,'wd classif')

cmd:option('-nThreads',6,'')
cmd:option('-iter_per_thread',8,'')
cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,nThreads=true,
                                        debug=true,iter_per_thread=true,
                                        color=false,layer=false,model=false,
                                        sbatch=false,rbatch=false,
                                      })
opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

opt.projname = opt.projname == '' and opt.name or opt.projname

cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Train logistic')

cutorch.setDevice(opt.gpu)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)


--------------------------------------------------------------------------------
-- useful variables
--------------------------------------------------------------------------------

local crop_size = 227

local batch_size = opt.rbatch + opt.sbatch

local n_batches = 100
local num_iter = 100
--------------------------------------------------------------------------------
-- [[
local load_models = paths.dofile 'load_models.lua'
model_base,image_transformer,feat_dim = load_models()

collectgarbage()
collectgarbage()

--------------------------------------------------------------------------------

local loadFeatsName, loadProjName, loadCalibName, getLinProj = paths.dofile 'load_utils.lua'

if opt.linproj then
  linear_proj = getLinProj()
  linear_proj:cuda()
  linear_proj:evaluate()
end

if opt.normalize then
  error('Need to check this path')
  normalizer = nnf.L2Normalize()
  model_base:add(normalizer)
end

model_base:cuda()
model_base:evaluate()

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)
--]]

--------------------------------------------------------------------------------
-- preparing dataset
--------------------------------------------------------------------------------

if opt.color == 'gray' then
  -- add a grayscale transformation before preprocessing
  image_transformer._preprocess = image_transformer.preprocess
  image_transformer.preprocess = function(self,I)
          I = image.rgb2y(I)
          return self:_preprocess(I)
        end
end

-- train
local imlist_synth = tds.Vec()
local imlist_synth_val = tds.Vec()
local synth_targets = tds.Vec()
local synth_targets_val = tds.Vec()
do
  print('Loading synthetic image names')
  local image_path
  if opt.name == 'mathieu_chairs' then
    image_path = '/media/francisco/rio/datasets/chairs_aubry/processed_tight'
    num_classes = 1 + 1
  elseif opt.name == 'ikeaobject' then
    image_path = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/crops/sphere_renders_perspective_object_new'
    num_classes = 11 + 1
  else
    error('Unrecognized dataset')
  end
  local image_list0 = {}
  for dir in paths.iterdirs(image_path) do
    table.insert(image_list0,dir)
  end
  table.sort(image_list0)
  local full_imlist = tds.Vec()
  local synth_classes = tds.Vec()
  -- select a percentage of the models for validation
  local num_train = math.ceil(#image_list0*0.85)
  for i,dir in ipairs(image_list0) do
    local names = {}
    for imfile in paths.iterfiles(paths.concat(image_path,dir)) do
      table.insert(names,imfile)
    end
    table.sort(names)
    local cls_type
    if opt.name == 'mathieu_chairs' then
      cls_type = 1
    elseif opt.name == 'ikeaobject' then
      cls_type = i
    end
    for k,v in pairs(names) do
      full_imlist:insert(paths.concat(image_path,dir,v))
      synth_classes:insert(cls_type+1) -- background
    end
  end
  local shuffle = torch.randperm(#full_imlist)
  -- select a percentage of the images for validation
  local num_train = math.ceil(#full_imlist*0.85)
  for i=1, num_train do
    imlist_synth:insert(full_imlist[shuffle[i] ])
    synth_targets:insert(synth_classes[shuffle[i] ])
  end
  for i=num_train+1,#full_imlist do
    imlist_synth_val:insert(full_imlist[shuffle[i] ])
    synth_targets_val:insert(synth_classes[shuffle[i] ])
  end
end

local imlist_real = tds.Vec()
do
  print('Loading real image names')
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
  local root_folder = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit/VOC2007/'
  local image_path = paths.concat(root_folder, 'JPEGImages')
  local imgset_path = paths.concat(root_folder,'ImageSets/Main/train.txt')
  local imlist_temp = lines_from(imgset_path)
  for k,v in ipairs(imlist_temp) do
    imlist_real:insert(paths.concat(image_path,v..'.jpg'))
  end
end


--------------------------------------------------------------------------------
-- creating data providers
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
-- create threads for loading the data
print('Creating threads for data loading')
local iter_per_thread = opt.iter_per_thread
local options = opt
local image_transf = image_transformer
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
donkeys = Threads(opt.nThreads,
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
                opt = options
                torch.setheaptracking(true)
                torch.manualSeed(idx)
                imagelist_real = imlist_real
                imagelist_synth = imlist_synth
                synth_targs = synth_targets
                nImages_real = #imagelist_real
                nImages_synth = #imagelist_synth
                local dt = matio.load('/home/francisco/work/libraries/rcnn/data/selective_search_data/voc_2007_train.mat')
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
                img_buffer = torch.FloatTensor(iter_per_thread,3,227,227)
                targets = torch.FloatTensor(iter_per_thread)
                iter_per_thread = iter_per_thread
                print(string.format('Starting donkey with id: %d', idx))
             end
              )

images = torch.FloatTensor(batch_size,3,227,227)
imtargets = torch.FloatTensor(batch_size)
function getBatch()
  imtargets:fill(1)
  for i=1,opt.rbatch/iter_per_thread do
    donkeys:addjob(
    function()
      -- load images
      for j=1,iter_per_thread do
        local imidx = torch.random(1,nImages_real)
        local I = image.load(imagelist_real[imidx],3,'float')
        I = image_transformer:preprocess(I)
        local bbidx = torch.random(1,roidb[imidx]:size(1))
        getCrop(img_buffer[j],I,roidb[imidx][bbidx])
      end
      collectgarbage()
      return img_buffer
    end,
    function(I)
      images:narrow(1,(i-1)*iter_per_thread+1,iter_per_thread):copy(I)
    end
    )
  end
  donkeys:synchronize()
  for i=1,opt.sbatch/iter_per_thread do
    donkeys:addjob(
    function()
      -- load images
      for j=1,iter_per_thread do
        local imidx = torch.random(1,nImages_synth)
        local I = image.load(imagelist_synth[imidx],3,'float')
        I = image_transformer:preprocess(I)
        local iH = I:size(2)
        local iW = I:size(3)
        local h1 = torch.random(1,iH/5)--iH-227
        local w1 = torch.random(1,iW/5)--iW-227
        local height = torch.random(iH/1.5,iH-h1)
        local width  = torch.random(iW/1.5,iW-w1)
        --getCrop(img_buffer[j],I,{w1,h1,w1+width,h1+height})
        targets[j] = synth_targs[imidx]
        img_buffer[j]:copy(image.scale(I[{{},{h1,h1+height},{w1,w1+width}}],
                                      crop_size,crop_size)
                          )
      end
      collectgarbage()
      return img_buffer, targets
    end,
    function(I,target)
      images:narrow(1,opt.rbatch+(i-1)*iter_per_thread+1,iter_per_thread):copy(I)
      imtargets:narrow(1,opt.rbatch+(i-1)*iter_per_thread+1,iter_per_thread):copy(target)
    end
    )
  end
  donkeys:synchronize()
  return images, imtargets
end

-------------------------------------------------------------------
-- validation data
-- --

-- validation

local function compute_features(images,batchSize, is_real)
  local input = torch.CudaTensor()
  local batch_images = torch.split(images,batchSize,1)
  local features = torch.FloatTensor()
  for i=1,#batch_images do
    input:resize(batch_images[i]:size()):copy(batch_images[i])
    local output = model_base:forward(input)
    if opt.linproj and is_real then
      output = linear_proj:forward(output)
    end
    if i==1 then
      features = output:float()
    else
      features = torch.cat(features,output:float(),1)
      collectgarbage()
    end
  end
  return features
end

-- real data
do
  local root_folder = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit/'
  local ds_test = nnf.DataSetPascal{
    image_set= 'val',
    classes=    {},
    datadir=    root_folder,
    roidbdir=   '/home/francisco/work/libraries/rcnn/data/selective_search_data',
    year=       2007
  }

  local feat_provider_test = nnf.RCNN(ds_test)
  feat_provider_test.image_transformer = image_transformer

  local div_fact = 10
  local iter_per_batch = math.ceil(#imlist_synth_val/opt.sbatch/div_fact) -- 54 -- to keep approx the same distribution as training, math.ceil(num_test/0.25/96)
  print('Batch provider validation: iter_per_batch = '..iter_per_batch)
  -- get background patches for testing
  local batch_provider_test = nnf.BatchProvider(feat_provider_test)
  batch_provider_test.nTimesMoreData = 10
  batch_provider_test.iter_per_batch = iter_per_batch
  batch_provider_test.fg_fraction = 0
  batch_provider_test.batch_size = opt.rbatch
  batch_provider_test.batch_dim = {3,crop_size,crop_size}
  batch_provider_test.do_flip = false

  batch_provider_test:setupData()
  real_feats_test = torch.FloatTensor()
  local bsize = 128

  for i=1,div_fact do --10
    print(i)
    local batches = batch_provider_test:getBatch()
    batches = batches:view(-1,3,227,227)
    if i==1 then
      real_feats_test = compute_features(batches,bsize,true)
    else
      real_feats_test = torch.cat(real_feats_test,compute_features(batches,bsize),1)
    end
  end
end

-- synthetic data
do
  print('Computing synthetic features for validation')
  local bsize = 128
  local num_synthetic_val = #imlist_synth_val
  local stval = torch.Tensor(num_synthetic_val)
  for i=1, num_synthetic_val do
    stval[i] = synth_targets_val[i]
  end
  synth_targets_val = nil
  synth_targets_val = stval
  synth_targets_val = synth_targets_val[{{1,num_synthetic_val}}]
  local imgs = torch.FloatTensor(num_synthetic_val, 3, crop_size, crop_size)
  for k,v in ipairs(imlist_synth_val) do
    if k > num_synthetic_val then
      break
    end
    local I = image.load(v,3,'float')
    imgs[k] = image.scale(image_transformer:preprocess(I):float(),crop_size,crop_size)
  end
  synth_feats_test = compute_features(imgs,bsize,false)
end

collectgarbage()
--------------------------------------------------------------------------------
-- create model
--------------------------------------------------------------------------------
print('Creating model for training')
train_model = nn.Sequential():add(nn.Linear(feat_dim,num_classes)):cuda()
train_model_save = train_model:clone('weight','bias')
criterion = nn.CrossEntropyCriterion():cuda()
print(train_model)
optimState = {learningRate = opt.lr2, weightDecay = opt.wd2, momentum = 0.9,
              learningRateDecay = 0}

optimizer = nn.Optim(train_model,optimState)

local targets_cuda = torch.CudaTensor(batch_size):fill(1)
targets_cuda:narrow(1,opt.rbatch+1,opt.sbatch):fill(2)


--------------------------------------------------------------------------------
-- training functions
--------------------------------------------------------------------------------

function train()
  model_base:evaluate()
  train_model:training()
  local tt = torch.Timer()
  print('Training...')
  local err = 0
  local conf_mat = optim.ConfusionMatrix(num_classes)
  local input_data = torch.CudaTensor()
  for j=1,n_batches do
    getBatch()
    input_data:resize(images:size()):copy(images)
    targets_cuda:resize(imtargets:size()):copy(imtargets)
    local features = model_base:forward(input_data) 
    if opt.linproj then
      local ff = features:narrow(1,1,opt.rbatch)
      local outlin = linear_proj:forward(ff)
      ff:copy(outlin)
    end
    -- train
    local err0, output = optimizer:optimize(optim.sgd,features,targets_cuda,criterion,false)
    err = err + err0
    conf_mat:batchAdd(output,targets_cuda)
  end
  conf_mat:updateValids()
  print(('  Elapsed time: %.2fs'):format(tt:time().real))
  return err/n_batches, conf_mat
end

function evaluate()
  model_base:evaluate()
  train_model:evaluate()
  local timer = torch.Timer()
  print('Evaluating...')
  local err_1 = 0
  local err_2 = 0
  local conf_mat = optim.ConfusionMatrix(num_classes)
  local bs = 128
  local target = torch.CudaTensor(bs)
  local inputs = torch.CudaTensor(bs,feat_dim) --3,227,22
    -- real images
  local split1 = real_feats_test:split(bs,1)
  for k,v in pairs(split1) do
    inputs:resize(v:size()):copy(v)
    target:resize(v:size(1)):fill(1)
    local output = train_model:forward(inputs)
    err_1 = err_1 + criterion:forward(output,target)
    conf_mat:batchAdd(output,target)
  end
  -- synthetic images
  local split1 = synth_feats_test:split(bs,1)
  local split2 = synth_targets_val:split(bs,1)
  for k,v in pairs(split1) do
    inputs:resize(v:size()):copy(v)
    target:resize(v:size(1)):copy(split2[k])
    local output = train_model:forward(inputs)
    err_2 = err_2 + criterion:forward(output,target)
    conf_mat:batchAdd(output,target)
  end
  local total_err = (err_1 + err_2)/(real_feats_test:size(1) + synth_feats_test:size(1))
  err_1 = err_1/real_feats_test:size(1)
  err_2 = err_2/synth_feats_test:size(1)
  conf_mat:updateValids()
  target = nil
  inputs = nil
  collectgarbage()
  print(('  Elapsed time: %.2fs'):format(timer:time().real))
  return total_err, err_1, err_2, conf_mat
end


--------------------------------------------------------------------------------
-- let's play
--------------------------------------------------------------------------------

logger = optim.Logger(paths.concat(opt.rundir,'loss.log'))
logger:setNames{'% mean loss (train set)', '% mean loss (test set)',
                '% mean loss real (test set)', '% mean loss synt (test set)'}
logger.showPlot = false

logger_prec = optim.Logger(paths.concat(opt.rundir,'ap.log'))
logger_prec:setNames{'% AP (train set)', '% AP (test set)'}
logger_prec.showPlot = false

for i=1,num_iter do
  print(('Iteration: %d/%d'):format(i,num_iter))
  if i%40 == 0 then
    optimState.learningRate = optimState.learningRate/10
    optimizer:setParameters(optimState)
  end
  local err_t,cm_t = train()
  local err_v, err_v1, err_v2,cm_v = evaluate()
  print(('  Train error: %g\n  Test error: %g, Real error: %g, Synth error: %g'):format(err_t,err_v,err_v1,err_v2))
  logger:add{err_t, err_v,err_v1,err_v2}
  logger:style{'-','-','-','-'}
  logger:plot()
  print('Training')
  print(cm_t)
  print('Validation')
  print(cm_v)
  logger_prec:add{cm_t.totalValid, cm_v.totalValid}
  logger_prec:style{'-','-'}
  logger_prec:plot()
  collectgarbage()
  if i%5 == 0 then
    print('Saving snapshot of the model')
    torch.save(paths.concat(opt.rundir,'model.t7'),train_model_save)
  end
end

if true then
  torch.save(paths.concat(opt.rundir,'model.t7'),train_model_save)
end
