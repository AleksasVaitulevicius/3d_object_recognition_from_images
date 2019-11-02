require 'nnf'
require 'trepl'
matio = require 'matio'
require 'cutorch'

require 'optim'
paths.dofile('Optim.lua')

local expfolder = 'projection'

-- Comand line arguments

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_chairs','')
cmd:option('-color','rgb','')
cmd:option('-model','caffenet','')
cmd:option('-layer','conv5','')
cmd:option('-lr',1e0,'')
cmd:option('-wd',0.0005,'weight decay (l2 regularization)')
cmd:option('-dist','cosine','distance type (cosine or l2)')
cmd:option('-fluctuations',false,' dont set it, old')
cmd:option('-spatial_inv',false,'')
cmd:option('-diag',false,'force diagonal matrix')
cmd:option('-relu',false,'add relu in the end of the projection')
cmd:option('-bn',false,'add batch normalization after the projection')
cmd:option('-switch_order',false,'')
cmd:option('-per_position',false,'')
cmd:option('-conv_proj','','use more complex projection types')
cmd:option('-num_iter',40,'number of iterations')
cmd:option('-step_lr',15,'divide learning rate every step_lr')
cmd:option('-div_lr',10,'division factor of the learning rate')
cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

local rootfolder = os.getenv('CACHEFOLDERPATH')
assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,debug=true,
                                        color=false,layer=false,model=false,
                                        dist=false,
                                      })
opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Learn projection')

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

local addRandomBG = paths.dofile 'load_images.lua'

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

local loadFeatsName = paths.dofile 'load_utils.lua'

do
  require 'hdf5'
  local fname = loadFeatsName()
  print('Reading features from '..fname)
  local f = hdf5.open(fname)
  local feats = f:read('/'):all()
  f:close()

  collectgarbage()
  collectgarbage()

  local cc = 0
  local dim = {}
  -- getting cad names in alphabetical order
  local cad_names = {}
  for i in pairs(feats) do
    cc = cc + 1
    table.insert(cad_names,i)
  end
  table.sort(cad_names)
  for i in pairs(feats) do dim = feats[i].features:size(); break; end
  n_elems = cc
  n_renders_cad = dim[1]--62 -- there are 62 renders per CAD
  features_1 = torch.FloatTensor(cc,n_renders_cad,table.unpack(feat_dim))
  features_2 = torch.FloatTensor(cc,n_renders_cad,table.unpack(feat_dim))
  local elm1 = 1
  local elm2 = 2
  if opt.switch_order then
    elm1 = 2
    elm2 = 1
  end
  for cc,i in ipairs(cad_names) do
    local ff_t = feats[i].features:transpose(1,2)
    features_1[cc]:copy(ff_t[elm1])
    features_2[cc]:copy(ff_t[elm2])
  end

  n_elems_cad = n_elems*n_renders_cad

  local num_train = math.floor(n_elems*0.85)*n_renders_cad--1184*62 -- ~85% of training data, without seeing the same instance

  if opt.spatial_inv then
    feats = nil
    num_train = num_train*6*6
    n_elems_cad = n_elems_cad*6*6
    features_1 = features_1:transpose(3,4):transpose(4,5):contiguous()
    collectgarbage()
    features_2 = features_2:transpose(3,4):transpose(4,5):contiguous()
    collectgarbage()
    collectgarbage()
  end

  features_1 = features_1:view(n_elems_cad,-1)
  features_2 = features_2:view(n_elems_cad,-1)

  features_1_test = features_1[{{num_train+1,-1},{}}]
  features_2_test = features_2[{{num_train+1,-1},{}}]

  features_1 = features_1[{{1,num_train},{}}]
  features_2 = features_2[{{1,num_train},{}}]

  feats = nil

  assert(features_1:size(2) == feat_dim[1], 'Something went wrong with the dimensions')
  collectgarbage()
  collectgarbage()
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
  error('Sure you want to add it ? Comment this then')
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
getBatchSequential = getBatchSequentialBase()


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
getBatchFluctuations = getBatchFluctuationsBase()



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
  local n = math.ceil(features_1:size(1)/batch_size)
  for i=1,n do
    if opt.fluctuations then
      inputs,targets = getBatchFluctuations()
    else
      inputs,targets = getBatchSequential()
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
--logger = optim.Logger(paths.concat(opt.rundir,'train.log'))
--logger:setNames{'% mean loss (train set)', '% mean loss (test set)'}
--logger.showPlot = false


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
  --logger:add{err, err_v}
  --logger:style{'-','-'}
  --logger:plot()
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
