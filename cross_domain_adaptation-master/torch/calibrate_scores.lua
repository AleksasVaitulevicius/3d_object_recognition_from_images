require 'nnf'
require 'trepl'
matio = require 'matio'
require 'hdf5'
require 'cutorch'

matio.use_lua_strings = true

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_chairs','')
cmd:option('-projname','','')
cmd:option('-color','rgb','')
cmd:option('-model','caffenet','')
cmd:option('-layer','conv5','')
cmd:option('-lr',1e0,'')
cmd:option('-wd',0.0005,'')
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
cmd:option('-synth_with_bg',false,'')

cmd:option('-crandom',false,'completely random patches sampled')
cmd:option('-linproj',true,'')
cmd:option('-normalize',true,'')
cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

local rootfolder = os.getenv('CACHEFOLDERPATH')
assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

local expfolder = 'calibration'

local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,debug=true,
                                        color=false,layer=false,model=false,
                                        dist=false,normalize=false,
                                      })
opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

opt.projname = opt.projname == '' and opt.name or opt.projname

cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Calibrate')

cutorch.setDevice(opt.gpu)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

torch.setheaptracking(true)

--------------------------------------------------------------------------------

crop_size = 227



local load_models = paths.dofile 'load_models.lua'
model_test,image_transformer,feat_dim = load_models()

local addRandomBG, getRCNNCrop, PatchProvider = paths.dofile 'load_images.lua'

local loadFeatsName, loadProjName, loadCalibName, getLinProj = paths.dofile 'load_utils.lua'

  --local fname = loadFeatsName()

if opt.normalize then
  if opt.per_position then
    normalizer = nn.Sequential()
    -- need to reshape, then transpose then reshape again
    local transform_input = nn.Sequential()
    transform_input:add(nn.View(-1,feat_dim/36,6,6))
    transform_input:add(nn.Transpose({2,3},{3,4}))
    transform_input:add(nn.Copy(nil,nil,true)) -- to make contiguous for view
    transform_input:add(nn.View(-1, feat_dim/36))

    normalizer:add(transform_input) -- input is (n*36,feat_dim/36)
    normalizer:add(nnf.L2Normalize())
    -- now invert back the transformation
    local detransform_input = nn.Sequential()
    detransform_input:add(nn.View(-1,6,6,feat_dim/36))
    detransform_input:add(nn.Transpose({3,4},{2,3}))
    detransform_input:add(nn.Copy(nil,nil,true))
    detransform_input:add(nn.View(-1,feat_dim))
    normalizer:add(detransform_input)
  else
    normalizer = nnf.L2Normalize()
  end

  model_test:add(normalizer)
end


if opt.linproj and opt.switch_order == false then
  assert(opt.normalize, 'check normalize !')
  model_test:insert(getLinProj(),model_test:size())
end

model_test:cuda()
model_test:evaluate()

print('Model used for the real data:')
print(model_test)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------
if opt.crandom then
  cls = {}
else
  cls = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
         'cat','chair','cow','diningtable','dog','horse','motorbike',
         'person','pottedplant','sheep','sofa','train','tvmonitor'}
end

--------------------------------------------------------------------------------
-- Detection part
--------------------------------------------------------------------------------

if opt.color == 'gray' then
  -- add a grayscale transformation before preprocessing
  image_transformer._preprocess = image_transformer.preprocess
  image_transformer.preprocess = function(self,I)
          I = image.rgb2y(I)
          return self:_preprocess(I)
        end
end

--[[
local ds_fold = '/home/francisco/work/projects/object-detection.torch/datasets/VOCdevkit/'
ds_train = nnf.DataSetPascal{image_set= 'train',
                            classes=    cls,
                            datadir=    ds_fold,
                            roidbdir=   '/home/francisco/work/libraries/rcnn/data/selective_search_data',
                            year=       2012}

feat_provider = nnf.RCNN(ds_train)
feat_provider.image_transformer = image_transformer
feat_provider_val = nnf.RCNN(ds_val)
feat_provider_val.image_transformer = image_transformer


-- get background patches to calibrate the detector
if true then

batch_provider = nnf.BatchProvider(feat_provider)
batch_provider.nTimesMoreData = opt.crandom and 200 or 50
batch_provider.iter_per_batch = 50
batch_provider.fg_fraction = 0
batch_provider.batch_size = 128--*50
batch_provider.batch_dim = {3,227,227}
batch_provider.do_flip = true
batch_provider:setupData()

end
--]]

local super_batch_size = 128*50

local num_threads = os.getenv('NTHREADS')
if num_threads then
  num_threads = tonumber(num_threads)
end
patch_provider = PatchProvider(super_batch_size, num_threads)

--input_val,target_val = batch_provider_val:getBatch()
collectgarbage()

--------------------------------------------------------------------------------
-- get the negative features
--------------------------------------------------------------------------------

function compute_features(images,batchSize)
  local input = torch.CudaTensor()
  local batch_images = torch.split(images,batchSize,1)
  local features = torch.FloatTensor()
  for i=1,#batch_images do
    input:resize(batch_images[i]:size()):copy(batch_images[i])
    if i==1 then
      features = model_test:forward(input):float()
    else
      features = torch.cat(features,model_test:forward(input):float(),1)
      collectgarbage()
    end
  end
  return features
end


-- compute background features to calibrate
if true then
  local num_batch_load = 30
  features = torch.FloatTensor(super_batch_size*num_batch_load,feat_dim)
  local bsize = 64
  print('Reading random patches for calibration...')
  for i=1,num_batch_load do
    print(string.format('  Loading: %2d/%2d',i,num_batch_load))
    --local batches = batch_provider:getBatch()
    --batches = batches:view(-1,3,227,227)
    batches = patch_provider()
  --  features:narrow(1,(i-1)*num_batch_load,num_batch_load):copy(compute_features(batches,bsize))
    features:narrow(1,(i-1)*super_batch_size+1,super_batch_size):copy(compute_features(batches,bsize))
  
    collectgarbage()
  end
end
-- clean up
ds_train = nil
feat_provider = nil
batch_provider = nil
patch_provider = nil
collectgarbage()
--------------------------------------------------------------------------------
-- get the positive features and estimate the calibration
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- create synthetic features
--------------------------------------------------------------------------------
if opt.linproj and opt.switch_order == false then
  --removeLinProj()
  if opt.normalize then
    model_test:remove(model_test:size()-1)
  else
    model_test:remove(model_test:size())
  end
  print('model after removing proj layers')
  print(model_test)
end


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
  n_renders_cad = dim[1] -- 62 -- there are 62 renders per CAD
  features_1 = torch.FloatTensor(cc,n_renders_cad,feat_dim)
  local feat_idx_nb = 1
  if opt.synth_with_bg then
    feat_idx_nb = 2
  end
  for cc,i in ipairs(cad_names) do
    local ff_t = feats[i].features:transpose(1,2)
    features_1[cc]:copy(ff_t[feat_idx_nb])
  end

  n_elems_cad = n_elems*n_renders_cad

  synth_feats = features_1:view(n_elems_cad,-1)

  feats = nil
  collectgarbage()
  collectgarbage()

  -- save name order to text file (alphabetical now)
  if true then
    local f = io.open(paths.concat(opt.rundir,'feat_names.txt'),'w')
    for k,v in ipairs(cad_names) do
      f:write(v..'\n')
    end
    f:close()
  end

  if opt.linproj and opt.switch_order == true then
    local tmodel = getLinProj()
    tmodel:cuda()
    tmodel:evaluate()
    print('Projecting synthetic features')
    local synth_feats_splitted = synth_feats:split(256,1)
    local synth_f_cuda = torch.CudaTensor()
    for k,v in pairs(synth_feats_splitted) do
      local fs = v:size(1)
      synth_f_cuda:resize(v:size()):copy(v)
      local o = tmodel:forward(synth_f_cuda)
      synth_feats:narrow(1,(k-1)*256+1,fs):copy(o)
    end
    synth_f_cuda = nil
    collectgarbage()
    collectgarbage()
  end

  if opt.normalize then
    print('Normalizing synthetic features')
    local synth_feats_splitted = synth_feats:split(256,1)
    local synth_f_cuda = torch.CudaTensor()
    for k,v in pairs(synth_feats_splitted) do
      local fs = v:size(1)
      synth_f_cuda:resize(v:size()):copy(v)
      local o = normalizer:forward(synth_f_cuda)
      synth_feats:narrow(1,(k-1)*256+1,fs):copy(o)
    end
    synth_f_cuda = nil
    collectgarbage()
    collectgarbage()
  end

end

-- cleaning memory. Don't need model anymore
model_test = nil
normalizer = nil
collectgarbage()

print('Computing mean features')
feats_mean = features:mean(1):cuda()

if opt.per_position then
  batch_size = 4096*8
else
  batch_size = 4096*8--16
end


feats_cuda = torch.CudaTensor()
synth_feats_cuda = torch.CudaTensor()

--results = torch.FloatTensor(features:size(1))
results_cuda = torch.CudaTensor()

if opt.per_position then
  bsize = 200--100
else
  bsize = 500
end

results_sorted = torch.FloatTensor()
results_sorted_idx = torch.LongTensor()

results_sorted_cuda = torch.CudaTensor()

end_idx = synth_feats:size(1)
end_idx0 = math.ceil(end_idx/bsize)

if opt.per_position then
  a = torch.FloatTensor(synth_feats:size(1)*6*6):zero()
  b = torch.FloatTensor(synth_feats:size(1)*6*6):zero()
else
  a = torch.FloatTensor(synth_feats:size(1)):zero()
  b = torch.FloatTensor(synth_feats:size(1)):zero()
end

collectgarbage()


if opt.per_position then
  print('Preparing data to have right dimensions')
  synth_feats = synth_feats:view(-1,feat_dim/36,36):transpose(2,3):transpose(1,2) -- (36,n,feat_dim/36)
  print('synth_feats size')
  print(synth_feats:size())
  synth_feats = synth_feats:contiguous()
  collectgarbage()
  collectgarbage()
  features = features:view(-1,feat_dim/36,36):transpose(2,3):transpose(1,2) -- (36,n,feat_dim/36)
  print('features size')
  print(features:size())

  features = features:contiguous()
  collectgarbage()
  collectgarbage()

  feats_split = features:split(batch_size,2)
  results = torch.FloatTensor(features:size(2),bsize*36)
else
  feats_split = features:split(batch_size,1)
  results = torch.FloatTensor(features:size(1),bsize)
end




print('Calibrating features')
local tt = torch.Timer()
for i=1,math.ceil(end_idx/bsize) do
  xlua.progress((i-1)*bsize+1,end_idx)
  local localbsize = (i<end_idx0) and bsize or (end_idx-1)%bsize + 1 -- number of cad views to use here
  --synth_feats_cuda:resize(synth_feats:size(2),localbsize):copy(synth_feats:narrow(1,(i-1)*bsize+1,localbsize):t())

  tt:reset()
  if opt.per_position then
    local temp_synth_feat = synth_feats:narrow(2,(i-1)*bsize+1,localbsize)
    synth_feats_cuda:resize(36,localbsize,feat_dim/36)
    synth_feats_cuda:copy(temp_synth_feat)
    results:resize(features:size(2),localbsize*36)
  else
    local temp_synth_feat = synth_feats:narrow(1,(i-1)*bsize+1,localbsize)
    synth_feats_cuda:resize(localbsize,feat_dim):copy(temp_synth_feat)
    results:resize(features:size(1),localbsize)
  end
  --print('T1: '..tt:time().real)
  collectgarbage()
  tt:reset()
  for j=1,#feats_split do
    ff = feats_split[j]
    if opt.per_position then
      --feats_cuda:resize(36,ff:size(1),ff:size(2)/36):copy(ff:view(-1,ff:size(2)/36,36):transpose(2,3):transpose(1,2))
      feats_cuda:resize(ff:size()):copy(ff)
      results_cuda:resize(36,ff:size(2),localbsize)
      results_cuda:bmm(feats_cuda,synth_feats_cuda:transpose(2,3)) -- gives (36,ff:size(2), localbsize)
      results:narrow(1,(j-1)*batch_size+1,ff:size(2)):copy(results_cuda:transpose(1,2):transpose(2,3))
    else
      feats_cuda:resize(ff:size()):copy(ff)
      results_cuda:resize(ff:size(1),localbsize)
      results_cuda:mm(feats_cuda,synth_feats_cuda:t())
      results:narrow(1,(j-1)*batch_size+1,ff:size(1)):copy(results_cuda)
    end
  end
  --print('T2: '..tt:time().real)
  --torch.sort(results_sorted,results_sorted_idx,results,1,false);
  collectgarbage()

  if opt.per_position then
    results_sorted:resize(results:size())
    --torch.sort(results_sorted,results,1,false);
    -- [[
    local nnn1 = localbsize/2
    local nnn2 = 36*2
    local tempcudat = torch.CudaTensor(results:size(1))

    tt:reset()
    for j=1,nnn1 do
      -- fix for cutorch bug
      -- sorting dimension should be contiguous
      local rtemp = results:narrow(2,(j-1)*nnn2+1,nnn2):t()
      -- use feats_cuda as temporary buffer
      feats_cuda:resize(rtemp:size()):copy(rtemp)
      results_sorted_cuda:resizeAs(feats_cuda)
      --[[
      for jjj=1,nnn2 do
        tempcudat:copy(feats_cuda[{{},jjj}])
        torch.sort(results_sorted_cuda[{{},jjj}],tempcudat,1,false);
      end
      --]]
      torch.sort(results_sorted_cuda,feats_cuda,2,false);
      if j%1 == 0 then
        collectgarbage()
      end
      results_sorted:narrow(2,(j-1)*nnn2+1,nnn2):copy(results_sorted_cuda:t())
    end
    --print('T3: '..tt:time().real)
    --]]
    tt:reset()
    s1 = results_sorted[torch.round(results:size(1)*0.9999)]:float()

    --local sfct = synth_feats_cuda:transpose(1,2)--:transpose(2,3) -- (localbsize,feat_dim/36,36)
    --temp_cuda_tensor_synth:resize(36,feat_dim/36,localbsize):copy(sfct)
    --s2 = (feats_mean:view(feat_dim/36,36):t()*temp_cuda_tensor_synth:t())[1]:float()--[1]
    local fmean = feats_mean:view(feat_dim/36,1,36):transpose(1,3) -- (36,1,feat_dim/36)
    local synthf = synth_feats_cuda:transpose(2,3) -- (36,feat_dim/36,localbsize)
    local tres = torch.bmm(fmean,synthf) -- (36,1,localbsize)
    --s2 = (torch.bmm(fmean,synthf))[1]:float()--[1]
    s2 = (tres:transpose(1,3)):float():view(-1)--[1]
    for j=1,localbsize*36 do
      a[(i-1)*bsize*36+j]=1/(s1[j]-s2[j]);
      b[(i-1)*bsize*36+j]=-a[(i-1)*bsize+j]*s2[j]-1;
    end
    print('T4: '..tt:time().real)
  else
    torch.sort(results_sorted,results,1,false);
    s1 = results_sorted[torch.round(results:size(1)*0.9999)]:float()
    s2 = (feats_mean*synth_feats_cuda:t())[1]:float()--[1]
    --a[i]=1/(s1-s2);
    --b[i]=-a[i]*s2-1;
    for j=1,localbsize do
      a[(i-1)*bsize+j]=1/(s1[j]-s2[j]);
      b[(i-1)*bsize+j]=-a[(i-1)*bsize+j]*s2[j]-1;
    end
  end
  collectgarbage()
end
print('Done!')

if true then
f = hdf5.open(paths.concat(opt.rundir,'calibration.h5'),'w')

local fname2 = ('/calibration/a')
f:write(fname2,a)
local fname2 = ('/calibration/b')
f:write(fname2,b)

f:close()

end
