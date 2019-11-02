require 'nnf'
require 'trepl'
matio = require 'matio'
require 'hdf5'
require 'cudnn'

matio.use_lua_strings = true

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_chairs','')
cmd:option('-projname','','')
cmd:option('-dataset','voc2012subset','dataset to do the evaluation')
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
cmd:option('-linproj',true,'')
cmd:option('-conv_proj','','')
cmd:option('-num_iter',40,'')
cmd:option('-step_lr',15,'')
cmd:option('-div_lr',10,'')
cmd:option('-normalize',true,'')
cmd:option('-subset',0,'0 == all data')
cmd:option('-subset_type','rand','')
cmd:option('-synth_with_bg',false,'')

cmd:option('-crandom',false,'completely random patches sampled')
cmd:option('-batchcomp',true,'')

cmd:option('-calibrate',true,'')
cmd:option('-boxes','','')
cmd:option('-best_k',100,'')
cmd:option('-ss','','')

cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

local rootfolder = os.getenv('CACHEFOLDERPATH')
assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

local expfolder = opt.dataset..'_detection_nn'
local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,debug=true,batchcomp=true,
                                        color=false,layer=false,model=false,
                                        dist=false,normalize=false,
                                        dataset=true,
                                      })
opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

opt.projname = opt.projname == '' and opt.name or opt.projname

cmd:log(opt.rundir .. '/log.log', opt)
cmd:addTime('Detect')

cutorch.setDevice(opt.gpu)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

local load_models = paths.dofile 'load_models.lua'
model_test,image_transformer,feat_dim = load_models()

paths.dofile 'load_images.lua'

local loadFeatsName, loadProjName, loadCalibName, getLinProj = paths.dofile 'load_utils.lua'

if opt.normalize then
  if opt.per_position then
    normalizer = nn.Sequential()
    -- need to reshape, then transpose then reshape again
    local transform_input = nn.Sequential()
    transform_input:add(nn.View(-1,feat_dim/36,6,6))
    transform_input:add(nn.Transpose({2,3},{3,4}))
    transform_input:add(nn.Copy(nil,nil,true)) -- to make contiguous for view
    transform_input:add(nn.View(-1, feat_dim/36))

    normalizer:add(transform_input)
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
else
  error('Check that!')
end

model_test:cuda()
model_test:evaluate()
print(model_test)

collectgarbage()
collectgarbage()

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------
-- create dataset
--------------------------------------------------------------------------------

ds = paths.dofile 'load_dataset.lua'

--------------------------------------------------------------------------------
-- create synthetic features
--------------------------------------------------------------------------------
do
  synth_feats = nil
  -- load if already computed
  if true then
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
    --features = torch.FloatTensor(cc/2,unpack(dim:totable()))
    n_elems = cc
    n_renders_cad = dim[1]--62 -- there are 62 renders per CAD
    features_1 = torch.FloatTensor(cc,n_renders_cad,feat_dim)
    --features_2 = torch.FloatTensor(cc,n_renders_cad,256,6,6)
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

    if opt.subset ~= 0 then
      error('Not sure if you want to use subsets. If yes, ask me and I will help you')
      if opt.subset_type == 'rand' then
        subset_perm = torch.randperm(n_elems_cad):long()
      elseif opt.subset_type == 'model' then
        subset_perm = torch.range(1,opt.subset):long()
      elseif opt.subset_type == 'select_view_mathieu_1' then
        -- images selected which faces the front of the object
        --19-30
        --50-61
        subset_perm = torch.LongTensor(n_elems*24):zero()
        local count = 0
        for i=1,n_elems do
          for ii = 1,12 do
            count = count + 1
            subset_perm[count] = 18 + ii + (i-1)*n_renders_cad
          end
          for ii = 1,12 do
            count = count + 1
            subset_perm[count] = 49 + ii + (i-1)*n_renders_cad
          end
        end
      elseif opt.subset_type == 'modelnet_handselect_m56' then
        local txtfile = '/home/francisco/work/projects/cross_domain/datasets/modelnet_chairs/handselected_56.txt'
        local mlines = {}
        for line in io.lines(txtfile) do 
          --table.insert(mlines,line)
          mlines[line] = true
        end
        local keep_idxs = {}
        local ecount = 0
        for i in pairs(feats) do
          ecount = ecount + 1
          if mlines[i] == true then
            for ii=1,72 do
              table.insert(keep_idxs, (ecount-1)*72 + ii)
            end
          end
        end
        print('Selected '.. ecount.. ' models and '.. #keep_idxs..' images.')
        subset_perm = torch.LongTensor(keep_idxs)

      elseif opt.subset_type == 'modelnet_handselect_v10' then
        local vangles = {1,4,7,10,15,19,23,28,31,34}
        --{36, 39, 42, 45, 48, 51, 54, 57, 60, 63}
        local keep_idxs = {}
        local ecount = 0
        for i in pairs(feats) do
          ecount = ecount + 1
          for _,ii in ipairs(vangles) do
            table.insert(keep_idxs, (ecount-1)*72 + ii + 36)
          end
        end
        subset_perm = torch.LongTensor(keep_idxs)
      end

      subset_perm = subset_perm[{{1,opt.subset}}]
      synth_feats = synth_feats:index(1,subset_perm)
      --matio.save('subset_perm.mat',{subset_perm=subset_perm})
      --error('Stop here')
    end

    feats = nil
    collectgarbage()
    collectgarbage()

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

  -- add features to network to compute the dot product
  if not opt.batchcomp then
    assert(not opt.per_position,[[haven't checked this path]])
    output_size = synth_feats:size(1)
    local linear = nn.Linear(synth_feats:size(2),output_size)
    linear.weight:copy(synth_feats)
    linear.bias:zero()
    model_test:add(linear)
  elseif not opt.per_position then
    -- to have the same number of elements per batch, prime factors of 1393*62
    -- are 2, 7, 31, 199
    -- prime factors of modelnet 1261*72 = 2,2,2,3,3,13,97
    --n_times_batch = opt.name == 'check_mathieu' and 31 or 18--14 et 18
    n_times_batch = true and 31 or 18--14 et 18
    n_total_synth_images = synth_feats:size(1)--86366
    output_size = n_total_synth_images/n_times_batch
    assert(n_total_synth_images % n_times_batch == 0, 'Ops, number of elements is not divisible by batch size')
    local linear = nn.Linear(synth_feats:size(2),output_size)
    --linear.weight:copy(synth_feats)
    linear.bias:zero()
    model_test:add(linear)
  else
    -- we wont add the synthetic data to the network, as it's not really
    -- a linear layer anymore. Will do it by hand
    -- don't do anything then, just define some useful variables
    n_times_batch = opt.name == 'mathieu_chairs' and 31 or 18--14 et 18
    n_total_synth_images = synth_feats:size(1)--86366
    output_size = n_total_synth_images/n_times_batch
    assert(n_total_synth_images % n_times_batch == 0, 'Ops, number of elements is not divisible by batch size')
    model_test:add(nn.Identity())
  end
  print(model_test)
  model_test:cuda()
  model_test:evaluate()
end

-- synthetic data does not have projections, only real
if opt.linproj and opt.switch_order == false then
  local insert_idx = model_test:size()
  if opt.normalize then
    insert_idx = insert_idx - 1
  end
  model_test:insert(getLinProj(),insert_idx)

  model_test:cuda()
  model_test:evaluate()
end


print(model_test)

-- optimization
--
if opt.normalize and opt.per_position then
  -- remove normalization and add a more optimized one for the real images
  model_test:remove(model_test:size()-1)
  normalizer = nn.Sequential()
  -- need to reshape, then transpose then reshape again
  local transform_input = nn.Sequential()
  transform_input:add(nn.View(-1,feat_dim/36,36))
  transform_input:add(nn.Transpose({2,3},{1,2}))
  transform_input:add(nn.Copy(nil,nil,true)) -- to make contiguous for view
  transform_input:add(nn.View(-1, feat_dim/36))

  normalizer:add(transform_input)
  normalizer:add(nnf.L2Normalize())
  -- now invert back the transformation
  local detransform_input = nn.Sequential()
  detransform_input:add(nn.View(36,-1,feat_dim/36))
  --detransform_input:add(nn.Transpose({3,4},{2,3}))
  --detransform_input:add(nn.Copy(nil,nil,true))
  --detransform_input:add(nn.View(-1,feat_dim))
  normalizer:add(detransform_input)
  model_test:add(normalizer)
  model_test:cuda()
  model_test:evaluate()
  print('Adding optimized normalization for real images')
  print(model_test)
end

if opt.per_position then
  -- do the reshaping here to make things faster overall
  synth_feats = synth_feats:view(-1,feat_dim/36,36):transpose(1,3) -- (36,feat_dim/36,n) -- (36,n,feat_dim/36)
  print('synth_feats size')
  print(synth_feats:size())
  synth_feats = synth_feats:contiguous()
  collectgarbage()
end

--------------------------------------------------------------------------------
-- features
--------------------------------------------------------------------------------
if opt.color == 'gray' then
  -- add a grayscale transformation before preprocessing
  image_transformer._preprocess = image_transformer.preprocess
  image_transformer.preprocess = function(self,I)
          I = image.rgb2y(I)
          return self:_preprocess(I)
        end
end

batch_size = 32--128
feat_provider = nnf.RCNN(ds)
feat_provider.image_transformer = image_transformer

--feat_provider = nnf.RCNN{dataset=ds,image_transformer=image_transformer,crop_size=227}

input = torch.FloatTensor()
input_split = {}
input_cuda = torch.CudaTensor()
feats = torch.FloatTensor()

if opt.calibrate then
  local fname = loadCalibName()
  print('Loading calibration file: '.. fname)
  local f = hdf5.open(fname,'r')
  a = f:read('/calibration/a'):all()
  b = f:read('/calibration/b'):all()
  f:close()

  if opt.subset ~= 0 then
    a = a:index(1,subset_perm)
    b = b:index(1,subset_perm)
  end

end

if opt.per_position then
  a = a:view(-1,1,36):transpose(1,3):cuda()
  b = b:view(-1,1,36):transpose(1,3):cuda()
  collectgarbage()
end

f = hdf5.open(paths.concat(opt.rundir,'results.h5'),'w')
options = hdf5.DataSetOptions()
options:setChunked(32, 32)
options:setDeflate()
options2 = hdf5.DataSetOptions()
options2:setChunked(32,1)
options2:setDeflate()
local end_idx =   ds:size()
local ttimer = torch.Timer()
local ttimer2 = torch.Timer()
local ttimer3 = torch.Timer()
if opt.batchcomp then
  local seq_temp = nn.Sequential()
  for i=1,model_test:size()-1 do
    seq_temp:add(model_test:get(i))
  end
  model = nn.Sequential()
  model:add(seq_temp)
  model:add(model_test:get(model_test:size()))
end
for i=1,end_idx do
  ttimer:reset()
  io.write(('Image %d/%d...'):format(i,end_idx))
  input:resize(ds.roidb[i]:size(1),3,227,227)
  --feats:resize(ds.roidb[i]:size(1),9216)
  if not opt.batchcomp then
    feats:resize(ds.roidb[i]:size(1),output_size)
  else
    feats:resize(ds.roidb[i]:size(1),output_size*n_times_batch)
  end
  for j=1,ds.roidb[i]:size(1) do
    input[j]:copy(feat_provider:getFeature(i,ds.roidb[i][j]:totable()))
  end
  input_split = input:split(batch_size,1)
  ttimer2:reset()
  for j,f in pairs(input_split) do
    input_cuda:resize(f:size())
    input_cuda:copy(f)
    if not opt.batchcomp then
      feats:narrow(1,(j-1)*batch_size+1,f:size(1)):copy(model_test:forward(input_cuda))
    elseif not opt.per_position then
      -- too many synthetic data, need to split and copy to the gpu every time
      local output_int = model:get(1):forward(input_cuda)
      local step_fact = n_total_synth_images/n_times_batch
      for nn = 1,n_times_batch do
        local sff = synth_feats:narrow(1,(nn-1)*step_fact+1,step_fact)
        model_test:get(model_test:size()).weight:copy(sff)
        local ff1 = feats:narrow(1,(j-1)*batch_size+1,f:size(1))
        local ff2 = ff1:narrow(2,(nn-1)*step_fact+1,step_fact)
        ff2:copy(model:get(2):forward(output_int))
      end
    else -- per spatial position
      local output_int = model_test:forward(input_cuda) -- (36,n,feat_dim/36)
      -- synth_feats is (36,feat_dim/36,nsynth), so just do bmm
      cuda_buffer1 = cuda_buffer1 or torch.CudaTensor(36,feat_dim/36,output_size)
      cuda_buffer2 = cuda_buffer2 or torch.CudaTensor(36,f:size(1),output_size) -- input:size(1)
      cuda_buffer2:resize(36,f:size(1),output_size)
      local step_fact = n_total_synth_images/n_times_batch
      for nn = 1,n_times_batch do
        cuda_buffer1:resize(36,feat_dim/36,output_size)
        cuda_buffer1:copy(synth_feats:narrow(3,(nn-1)*step_fact+1,step_fact))
        cuda_buffer2:bmm(output_int,cuda_buffer1) -- dot product between real/synthetic
        -- adding calibration
        -- a and b are (36,1,nsynth)
        local a_t = a:narrow(3,(nn-1)*step_fact+1,step_fact)
        local b_t = b:narrow(3,(nn-1)*step_fact+1,step_fact)
        cuda_buffer2:cmul(a_t:expandAs(cuda_buffer2))
        cuda_buffer2:add(b_t:expandAs(cuda_buffer2))
        -- relu
        cuda_buffer2:cmax(0) -- (36,n,nsynth)
        -- sum and copy to results
        cuda_buffer1:sum(cuda_buffer2,1)
        local ff1 = feats:narrow(1,(j-1)*batch_size+1,f:size(1))
        local ff2 = ff1:narrow(2,(nn-1)*step_fact+1,step_fact)
        ff2:copy(cuda_buffer1)
      end
    end
  end
  io.write((' end feats %.1f s'):format(ttimer2:time().real))
  ttimer3:reset()
  if opt.calibrate and not opt.per_position then
    feats:cmul(a:view(1,-1):expandAs(feats))
    feats:add(b:view(1,-1):expandAs(feats))
  end
  io.write((' end calib %.1f s'):format(ttimer3:time().real))
--
  if opt.best_k > 0 then
    local ff,f_idx = feats:sort(2,true) -- get best elements per bbox
    ff    = ff[{{},{1,opt.best_k}}]
    f_idx = f_idx[{{},{1,opt.best_k}}]
    local fname  = ('/%s/distances'):format(ds:getName(i))
    local fname2 = ('/%s/indexes'):format(ds:getName(i))
    f:write(fname,ff,options)
    f:write(fname2,f_idx,options)
  else
    local fname = ('/%s/distances'):format(ds:getName(i))
    f:write(fname,feats,options)
  end
  collectgarbage()
  io.write((' Done in %.1f s\n'):format(ttimer:time().real))
  print(('Image %d/%d'):format(i,end_idx))
end
f:close()
