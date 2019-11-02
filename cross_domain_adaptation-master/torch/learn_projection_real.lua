require 'cunn'
require 'cudnn'
require 'nnf'
require 'trepl'
require 'optim'
matio = require 'matio'
require 'loadcaffe'
paths.dofile('../Optim.lua')

local rootfolder = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/projects/cross_domain/cachedir/'
local expfolder = 'projection'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_real','')
cmd:option('-color','rgb','')
cmd:option('-model','caffenet','')
cmd:option('-layer','conv5','')
cmd:option('-lr',1e0,'')
cmd:option('-wd',0.0005,'')
cmd:option('-dist','cosine','')
cmd:option('-fluctuations',false,'')
cmd:option('-spatial_inv',false,'')
cmd:option('-diag',false,'')
cmd:option('-relu',false,'')
cmd:option('-bn',false,'')
cmd:option('-switch_order',false,'')
cmd:option('-per_position',false,'')
cmd:option('-conv_proj','','')
cmd:option('-num_iter',40,'')
cmd:option('-step_lr',15,'')
cmd:option('-div_lr',10,'')
cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,debug=true,
                                        color=false,layer=false,model=false,
                                        dist=false,
                                      })
opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

cmd:log(opt.rundir .. '/log.log', opt)
cmd:addTime('Save features')

cutorch.setDevice(opt.gpu)

--assert(opt.spatial_inv and (opt.layer == 'conv4' or opt.layer == 'conv5'))

--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------
local load_models = paths.dofile 'load_models.lua'
model_test,image_transformer,feat_dim = load_models()

paths.dofile 'load_images.lua'

--------------------------------------------------------------------------------

input = {torch.CudaTensor()}

function compute_fc7(images,batchSize)
  local batch_images = torch.split(images,batchSize,1)
  local features = torch.FloatTensor()
  for i=1,#batch_images do
    input[1]:resize(batch_images[i]:size()):copy(batch_images[i])
    if i==1 then
      features = model_test:forward(input[1]):float()
    else
      features = torch.cat(features,model_test:forward(input[1]):float(),1)
      collectgarbage()
    end
  end
  return features
end

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------

if opt.spatial_inv then
  feat_dim = {feat_dim/36,6,6}
else
  feat_dim = {feat_dim}
end

batch_size = opt.conv_proj == 'fcRfc' and 32 or 128
inputs = {torch.CudaTensor(batch_size,table.unpack(feat_dim)),
          torch.CudaTensor(batch_size,table.unpack(feat_dim))}
targets = torch.CudaTensor(batch_size):zero()

--------------------------------------------------------------------------------
-- load pre-computed features
--------------------------------------------------------------------------------


do
  print('Loading images')
  local imgpath = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/mathieu_aligned'
  local image_list0 = {}
  for dir in paths.iterdirs(imgpath) do
    table.insert(image_list0,dir)
  end
  table.sort(image_list0)
  local image_list_real = {}
  local image_list_synth = {}
  local image_list = {}
  for i,dir in ipairs(image_list0) do
    local names = {}
    for imfile in paths.iterfiles(paths.concat(imgpath,dir)) do
      table.insert(names,imfile)
    end
    table.sort(names)
    for j,n in ipairs(names) do
      if j%2 == 1 then
        table.insert(image_list_real,paths.concat(imgpath,dir,n))
      else
        table.insert(image_list_synth,paths.concat(imgpath,dir,n))
      end
    end
  end
  for i = 1,#image_list_real do
    image_list[i] = {real=image_list_real[i],synth=image_list_synth[i]}
  end
  local num_images = #image_list*2 -- add flipped version as well
  local images = torch.FloatTensor(num_images,2,3,256,256)
  local shuffle = torch.randperm(num_images)
  local I
  for k,v in ipairs(image_list) do
    -- first synthetic images
    I = image.load(v.synth)
    I = image_transformer:preprocess(I):float()
    images[shuffle[(k-1)*2+1] ][1] = I 
    I = image.hflip(I)
    images[shuffle[k*2] ][1] = I
    -- then real images
    I = image.load(v.real)
    I = image_transformer:preprocess(I):float()
    images[shuffle[(k-1)*2+1] ][2] = I
    I = image.hflip(I)
    images[shuffle[k*2] ][2] = I 
  end
--   
  local num_train = math.floor(num_images*0.85)
  local num_test = num_images-num_train
--
  images_train = images:narrow(1,1,num_train):clone()
  images_test  = images:narrow(1,num_train+1,num_test):clone()
--  
  images = nil;
  collectgarbage()

  features_1_test = torch.FloatTensor(num_test,feat_dim[1])
  features_2_test = torch.FloatTensor(num_test,feat_dim[1])

  local images_test_flat = images_test:view(-1,3,256,256)
  local images_test_split = images_test_flat:split(64,1)

  print('Computing test features')
  local temp_data = torch.FloatTensor(64,3,227,227)
  local temp_data_cuda = torch.CudaTensor(64,3,227,227)
  for k,v in ipairs(images_test_split) do
    local lnum_imgs = v:size(1)
    temp_data:resize(lnum_imgs,3,227,227)
    for i=1,lnum_imgs do
      temp_data[i]:copy(image.scale(v[i],227,227))
    end
    temp_data_cuda:resize(temp_data:size()):copy(temp_data)
    local output = model_test:forward(temp_data_cuda)
    for i=1,lnum_imgs/2 do
      features_1_test[(k-1)*32+i]:copy(output[i*2-1])
      features_2_test[(k-1)*32+i]:copy(output[i*2])
    end
  end
  
  images_test = nil
  temp_data_cuda = nil
  temp_data = nil
  collectgarbage()

  assert(opt.spatial_inv == false, 'Not supported')

end

if opt.fluctuations then
  paths.dofile 'sample_synthetic_images.lua'
  synthetic_sampler = sample_synthetic_data(batch_size,nil,false)
end

--------------------------------------------------------------------------------
-- define model and criterion
--------------------------------------------------------------------------------
model = nn.Sequential()
prl = nn.ParallelTable()

local idmod = nn.Sequential()
idmod:add(nn.Identity())

if (opt.relu or opt.bn or opt.per_position) and (opt.conv_proj ~= '') then
  error('Mixture not supported')
end

local seq = nn.Sequential()
if opt.diag then
  seq:add(nn.CMul(feat_dim[1]))
  seq:add(nn.Add(feat_dim[1]))
elseif opt.conv_proj == '' then
  seq:add(nn.Linear(feat_dim[1],feat_dim[1]))
elseif opt.conv_proj == '1x1' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,1,1,1,1))
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == '3x3' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,3,3,1,1,1,1))
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == '2t1x1' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,1,1,1,1))
  seq:add(nn.ReLU())
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,1,1,1,1))
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == '2t3x3' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,3,3,1,1,1,1))
  seq:add(nn.ReLU())
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,3,3,1,1,1,1))
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == '1x1relu' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,1,1,1,1))
  seq:add(nn.ReLU())
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == '3x3relu' then
  seq:add(nn.View(-1,feat_dim[1]/36,6,6))
  seq:add(nn.SpatialConvolutionMM(feat_dim[1]/36,feat_dim[1]/36,3,3,1,1,1,1))
  seq:add(nn.ReLU())
  seq:add(nn.View(-1):setNumInputDims(3))
elseif opt.conv_proj == 'fcRfc' then
  seq:add(nn.Linear(feat_dim[1],feat_dim[1]))
  seq:add(nn.ReLU(true))
  seq:add(nn.Linear(feat_dim[1],feat_dim[1]))
end
if opt.bn then
  seq:add(nn.BatchNormalization(feat_dim[1]))
end
if opt.relu then
  seq:add(nn.ReLU())
end

if opt.per_position then
  -- need to reshape, then transpose then reshape again
  local transform_input = nn.Sequential()
  transform_input:add(nn.View(-1,feat_dim[1]/36,6,6))
  transform_input:add(nn.Transpose({2,3},{3,4}))
  transform_input:add(nn.Copy(nil,nil,true)) -- to make contiguous for view
  transform_input:add(nn.View(-1, feat_dim[1]/36))
  --transform_input:add(nn.Reshape(batch_size*6*6, feat_dim[1]/36)) -- reshape is crap :)
  -- add it to both branches
  idmod:add(transform_input:clone())
  seq:add(transform_input:clone())
end

prl:add(idmod)
prl:add(seq)

model:add(prl)
if opt.dist == 'cosine' then
  model:add(nn.CosineDistance())
  model:add(nn.AddConstant(-1))
elseif opt.dist == 'l2' then
  model:add(nn.PairwiseDistance(2))
end

model:cuda();
print(model)

--parameters,gradParameters = model:getParameters()

divide_loss = true;

criterion = nn.MSECriterion():cuda()
criterion.sizeAverage = false

optimState = {learningRate = opt.lr, weightDecay = opt.wd, momentum = 0.9,
              learningRateDecay = 0}

--------------------------------------------------------------------------------
-- define functions batch provider
--------------------------------------------------------------------------------

function getBatch()
  local p = torch.randperm(features_1:size(1))
  targets:resize(batch_size):zero()
  for i=1,batch_size do
    inputs[1][i]:copy(features_1[p[i]])
    inputs[2][i]:copy(features_2[p[i]])
  end
  return inputs,targets
end

function getBatchSequentialBase()
  local dd1 = features_1:split(batch_size,1)
  local dd2 = features_2:split(batch_size,1)
  local i = 1
  return function()
      inputs[1]:resize(dd1[i]:size()):copy(dd1[i])
      inputs[2]:resize(dd2[i]:size()):copy(dd2[i])
      if opt.per_position then
        targets:resize(dd1[i]:size(1)*36):zero()
      else
        targets:resize(dd1[i]:size(1)):zero()
      end

      i = (i % #dd1) + 1
    return inputs,targets
  end
end
--getBatchSequential = getBatchSequentialBase()


function getBatchFluctuationsBase()
  local mask_tol = 240/255
  local synth_img_cuda = torch.CudaTensor(batch_size,3,227,227)
  local synth_img_bg = torch.FloatTensor(batch_size,3,227,227)
  local function retf()
    local synth_imgs = synthetic_sampler()
    for i=1,batch_size do
      synth_img_bg[i] = addRandomBG(synth_imgs[i],mask_tol)
      synth_imgs[i] = image_transformer:preprocess(synth_imgs[i])
    end
    synth_img_cuda:copy(synth_imgs)
    local output_synth = model_test:forward(synth_img_cuda)
    inputs[1]:resize(output_synth:size()):copy(output_synth)
    synth_img_cuda:copy(synth_img_bg)
    local output_synth = model_test:forward(synth_img_cuda)
    inputs[2]:resize(output_synth:size()):copy(output_synth)
    targets:resize(batch_size):zero()
    return inputs, targets, synth_imgs, synth_img_bg
  end
  return retf
end
--getBatchFluctuations = getBatchFluctuationsBase()

function getBatchRealBase()
  local imgs_cuda = torch.CudaTensor(batch_size,3,227,227)
  local imgs = torch.FloatTensor(batch_size,3,227,227)
  local function retf()
    local shuffle = torch.randperm(images_train:size(1))
    for i=1,batch_size do
      imgs[i]:copy(randomCrop(images_train[shuffle[i]][1],{3,227,227}))
    end
    imgs_cuda:copy(imgs)
    local output_synth = model_test:forward(imgs_cuda)
    inputs[1]:resize(output_synth:size()):copy(output_synth)
    for i=1,batch_size do
      imgs[i]:copy(randomCrop(images_train[shuffle[i]][2],{3,227,227}))
    end
    imgs_cuda:copy(imgs)
    local output_synth = model_test:forward(imgs_cuda)
    inputs[2]:resize(output_synth:size()):copy(output_synth)
    targets:resize(batch_size):zero()
    return inputs, targets
  end
  return retf
end
getBatchReal = getBatchRealBase()


function evaluate()
  model:evaluate()
  local err_0 = 0
  local dd1 = features_1_test:split(batch_size,1)
  local dd2 = features_2_test:split(batch_size,1)
  for i=1,#dd1 do
    inputs[1]:resize(dd1[i]:size()):copy(dd1[i])
    inputs[2]:resize(dd2[i]:size()):copy(dd2[i])
    if opt.per_position then
      targets:resize(dd1[i]:size(1)*36):zero()
    else
      targets:resize(dd1[i]:size(1)):zero()
    end

    local outputs = model:forward(inputs)
    local f = criterion:forward(outputs,targets)
    if divide_loss then
      if opt.per_position then
        -- in this situation, the batch size is increased
        -- by 36, so should take that into account
        -- maybe dividing by outputs:size(1) is better ?
        f = f/(batch_size*6*6)
      else
        f = f/batch_size
      end
    end
    err_0 = err_0 + f/#dd1
  end
  return err_0
end

--------------------------------------------------------------------------------
optimizer = nn.Optim(model,optimState)

function train()
  model:training()
  -- do one epoch
  local err = 0
  local n = 500--math.ceil(features_1:size(1)/batch_size)
  for i=1,n do
    if opt.fluctuations then
      inputs,targets = getBatchFluctuations()
    else
      --inputs,targets = getBatchSequential()
      inputs,targets = getBatchReal()
    end

    --[[
    local feval = function(x)
        if x ~= parameters then
          parameters:copy(x)
        end
        gradParameters:zero()
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs,targets)
        local df_do = criterion:backward(outputs,targets)
        model:backward(inputs,df_do)
        if divide_loss then
          gradParameters:div(batch_size)
          f = f/batch_size
        end
        return f,gradParameters
      end
      local x,fx = optim.sgd(feval,parameters,optimState)
      err = err + fx[1]
      --]]
      -- using Optim from Facebook to take off the wd from bias easily
      err = err + optimizer:optimize(optim.sgd,inputs,targets,criterion)
    end
    err = err/n
    return err
end

--------------------------------------------------------------------------------
logger = optim.Logger(paths.concat(opt.rundir,'train.log'))
logger:setNames{'% mean loss (train set)', '% mean loss (test set)'}
logger.showPlot = false


err_val0 = evaluate()
print('Initial validation error: '..err_val0)

for i=1,opt.num_iter do
  print('Epoch: '..i)
  if i%opt.step_lr == 0 then --10
    optimState.learningRate = optimState.learningRate/opt.div_lr--10
    optimizer:setParameters(optimState)
  end
  local err = train()
  local err_v = evaluate()
  print(('  Train error: %g\n  Test error: %g'):format(err,err_v))
  logger:add{err, err_v}
  logger:style{'-','-'}
  logger:plot()
end


if opt.conv_proj == '' then
  matio.compression = matio.ffi.COMPRESSION_NONE
  local mat
  if opt.diag then
    local mm = model:get(1):get(2) -- sequential with cmul and add
    local ss = mm:get(1).weight:size(1)
    mat = torch.FloatTensor(ss+1,ss+1):zero()
    mat[{{1,ss},{1,ss}}]:copy(mm:get(1).weight:float():diag())
    mat[{{1,ss},-1}]:copy(mm:get(2).bias)
  else
    local mm = model:get(1):get(2):get(1)
    local ss = mm.weight:size(1)
    if opt.spatial_inv then
      local sdim = 6
      mat = torch.FloatTensor(ss*sdim*sdim+1,ss*sdim*sdim+1):zero()
      for id1 = 0, ss-1 do
        local bb = mm.bias[id1+1]
        for id2 = 0, ss-1 do
          local v = mm.weight[id1+1][id2+1]
          for id3 = 0,sdim*sdim-1 do
              mat[id1*sdim*sdim + id3 + 1][id2*sdim*sdim + id3 + 1] = v
              mat[{{id1*sdim*sdim + id3 + 1},{-1}}] = bb
          end
        end
      end
    else
      mat = torch.FloatTensor(ss+1,ss+1):zero()
      mat[{{1,ss},{1,ss}}]:copy(mm.weight)
      mat[{{1,ss},-1}]:copy(mm.bias)
    end
  end
  local fname = paths.concat(opt.rundir,'projection.mat')
  print('Saving projection matrix to '..fname)
  local datasave = {P=mat}
  if opt.bn then
    local mm = model:get(1):get(2):get(2)
    datasave.bnw = mm.weight:float()
    datasave.bnb = mm.bias:float()
    datasave.bnm = mm.running_mean:float()
    datasave.bns = mm.running_std:float()
  end
  matio.save(fname,datasave)
else
  local mm = model:get(1):get(2)
  local fname = paths.concat(opt.rundir,'projection.t7')
  print('Saving projection matrix to '..fname)
  torch.save(fname, mm)
end
--image.display(model:get(1):get(2).weight,0.1)
