require 'cunn'
require 'cudnn'
require 'nnf'
require 'trepl'
require 'optim'
matio = require 'matio'
require 'loadcaffe'
require 'hdf5'

matio.use_lua_strings = true


local rootfolder = os.getenv('CACHEFOLDERPATH')
assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

local expfolder = 'baseline_classifier_detection'

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
cmd:option('-relu',false,'')
cmd:option('-conv_proj','','')
cmd:option('-rbatch',96,'n_real_feats_per_batch')
cmd:option('-sbatch',32,'n_synth_feats_per_batch')
cmd:option('-synth_with_bg',false,'') -- fixed_bg
cmd:option('-nodamp',false,'')
cmd:option('-lr',1e0,'lr proj')
cmd:option('-wd',5e-4,'wd proj')
cmd:option('-lr2',0.001,'lr classif')
cmd:option('-wd2',5e-4,'wd classif')
cmd:option('-lr2_step',40,'lr classif step')
cmd:option('-lr2_div',10,'lr classif division factor')

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

cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Baseline detect')


cutorch.setDevice(opt.gpu)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------

local function loadClassifName()
  local rootfolder = os.getenv('CACHEFOLDERPATH')
  assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')

  local expfolder = 'baseline_classifier'

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-name','mathieu_chairs','')
  cmd:option('-projname','','')
  cmd:option('-color','rgb','')
  cmd:option('-model','caffenet','')
  cmd:option('-layer','conv5','')
  cmd:option('-normalize',false,'')
  cmd:option('-linproj',true,'')
  cmd:option('-relu',false,'')
  cmd:option('-conv_proj','','')
  cmd:option('-rbatch',96,'n_real_feats_per_batch')
  cmd:option('-sbatch',32,'n_synth_feats_per_batch')
  cmd:option('-synth_with_bg',false,'') -- fixed_bg
  cmd:option('-nodamp',false,'')
  cmd:option('-lr',1e0,'lr proj')
  cmd:option('-wd',5e-4,'wd proj')
  cmd:option('-lr2',0.001,'lr classif')
  cmd:option('-wd2',5e-4,'wd classif')
  cmd:option('-lr2_step',40,'lr classif step')
  cmd:option('-lr2_div',10,'lr classif division factor')

  cmd:option('-nThreads',6,'')
  cmd:option('-iter_per_thread',8,'')
  cmd:option('-debug',false,'')
  cmd:option('-gpu',1,'')

  local opt_temp = cmd:parse(arg or {})

  local savefolder = paths.concat(rootfolder,opt.name, expfolder)

  local fname = cmd:string(opt_temp.name, opt_temp, {gpu=true, name=true,nThreads=true,
                                          debug=true,iter_per_thread=true,
                                          color=false,layer=false,model=false,
                                          sbatch=false,rbatch=false,
                                        })
  local fname = paths.concat(savefolder,fname,'model.t7')
  assert(paths.filep(fname),'Classifier file could not be found ' .. fname)
  return fname
end


local load_models = paths.dofile 'load_models.lua'
model_test,image_transformer,feat_dim = load_models()

local loadFeatsName, loadProjName, loadCalibName, getLinProj = paths.dofile 'load_utils.lua'

if opt.linproj then
  linear_proj = getLinProj()
  model_test:add(linear_proj)
end

if opt.normalize then
  error('Check it')
  normalizer = nnf.L2Normalize()
  model_test:add(normalizer)
end

-- add classfier to the network
if true then
  local fname = loadClassifName()
  print('Loading classifier from: '..fname)
  local classifier = torch.load(fname)
  model_test:add(classifier)
end

model_test:add(nn.SoftMax())

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
local im_dir
local ss_path
local img_ext
if opt.name == 'mathieu_chairs' then
  im_dir = '/home/francisco/work/datasets/VOCdevkit/VOC2012/JPEGImages'
  ss_path = '/home/francisco/work/projects/cross_domain/torch_new/mathieu_chairs/mathieu_chairs_ss_bboxes.mat'
  img_ext = '.jpg'
elseif opt.name == 'ikeaobject' then
  im_dir = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/3d_toolbox_notfinal'
  ss_path = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/ikea_object/ss_boxes.mat'
  img_ext = ''
elseif opt.name == 'W-UG_chair' then
  im_dir = '/home/francisco/work/datasets/VOCdevkit/VOC2012/JPEGImages'
  ss_path = '/home/francisco/work/projects/cross_domain/torch_new/mathieu_chairs/mathieu_chairs_ss_bboxes.mat'
  img_ext = '.jpg'
end

do
  ds = {}
  ds.im_dir = im_dir
  local dt = matio.load(ss_path)
  ds.im_idx = dt.images
  -- get Image
  ds.getImage = function(self,i)
    return image.load(paths.concat(self.im_dir,self.im_idx[i]..img_ext))
  end
  -- dataset size
  function ds:size()
    return #self.im_idx
  end
  if opt.name == 'mathieu_chairs' then
    function ds:getName(i) return self.im_idx[i] end
  elseif opt.name == 'ikeaobject' then
    function ds:getName(i) return self.im_idx[i]:sub(10,-5) end
  elseif opt.name == 'W-UG_chair' then
    function ds:getName(i) return self.im_idx[i] end
  end
  -- bboxes
  function ds:loadROIDB()
    local dt = matio.load(ss_path)
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

if opt.name == 'mathieu_chairs' then
  output_size = 2
elseif opt.name == 'ikeaobject' then
  output_size = 12
elseif opt.name == 'W-UG_chair' then
  output_size = 2
end
batch_size = 32--128
feat_provider = nnf.RCNN(ds)
feat_provider.image_transformer = image_transformer

input = torch.FloatTensor()
input_split = {}
input_cuda = torch.CudaTensor()
feats = torch.FloatTensor()


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

for i=1,end_idx do
  ttimer:reset()
  io.write(('Image %d/%d...'):format(i,end_idx))
  input:resize(ds.roidb[i]:size(1),3,227,227)
  feats:resize(ds.roidb[i]:size(1),output_size)

  for j=1,ds.roidb[i]:size(1) do
    input[j]:copy(feat_provider:getFeature(i,ds.roidb[i][j]:totable()))
  end
  input_split = input:split(batch_size,1)
  ttimer2:reset()
  for j,f in pairs(input_split) do
    input_cuda:resize(f:size())
    input_cuda:copy(f)
    feats:narrow(1,(j-1)*batch_size+1,f:size(1)):copy(model_test:forward(input_cuda))
  end
  io.write((' end feats %.1f s'):format(ttimer2:time().real))

  local fname  = ('/%s/scores'):format(ds:getName(i))
  f:write(fname,feats,options2)

  collectgarbage()
  io.write((' Done in %.1f s\n'):format(ttimer:time().real))
  print(('Image %d/%d...'):format(i,end_idx))
end
f:close()
