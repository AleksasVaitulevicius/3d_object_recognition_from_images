require 'trepl'
require 'cutorch'

local rootfolder = os.getenv('CACHEFOLDERPATH')--'/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/projects/cross_domain/cachedir/'
image_path = os.getenv('CADIMGPATH')

rand_image_path = os.getenv('RANDIMGPATH')

assert(rootfolder,'need to set environment variable CACHEFOLDERPATH with the folder which will contain all the cache')
assert(image_path,'need to set environment variable CADIMGPATH with the folder which contains the cad models')
assert(rand_image_path,'need to set environment variable RANDIMGPATH with the folder which contains images to crop patches')

--[[
if opt.name == 'ikeaobject' then
  image_path = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/crops/sphere_renders_perspective_object_new'
elseif opt.name == 'ikea' then
  image_path = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/crops/sphere_renders_perspective_all'
end
--]]

local expfolder = 'features'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-name','mathieu_chairs','')
cmd:option('-color','rgb','rgb or gray')
cmd:option('-model','caffenet','')
cmd:option('-layer','conv5','')
cmd:option('-debug',false,'')
cmd:option('-gpu',1,'')

opt = cmd:parse(arg or {})

opt.rundir = cmd:string(opt.name, opt, {gpu=true, name=true,debug=true,
                                        color=false,layer=false,model=false,
                                      })

local savefolder = paths.concat(rootfolder,opt.name, expfolder)

opt.rundir = paths.concat(savefolder, opt.rundir)
paths.mkdir(opt.rundir)

cmd:log(opt.rundir .. '/log', opt)
cmd:addTime('Save features')

cutorch.setDevice(opt.gpu)

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------
local load_models = paths.dofile 'load_models.lua'
model_test,image_transformer,feat_dim = load_models()

collectgarbage()
collectgarbage()

torch.manualSeed(24011989)
cutorch.manualSeed(24011989)

--------------------------------------------------------------------------------

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

if not opt.debug then
  require 'hdf5'
  f = hdf5.open(paths.concat(opt.rundir,'features.h5'),'w')
  options = hdf5.DataSetOptions()
  options:setChunked(32, 32)
  options:setDeflate()
  options2 = hdf5.DataSetOptions()
  options2:setChunked(16,1,32, 32)
  options2:setDeflate()
  options3 = hdf5.DataSetOptions()
  options3:setChunked(8,2, 64)
  options3:setDeflate()
end

-- get cad folders
image_list = {}
for dir in paths.iterdirs(image_path) do
  table.insert(image_list,dir)
end
table.sort(image_list)
-- count renders per cad
local render_count
for m_idx,dir in ipairs(image_list) do
  local lrender_count = 0
  for imfile in paths.iterfiles(paths.concat(image_path,dir)) do
    lrender_count = lrender_count + 1
  end
  if not render_count then
    render_count = lrender_count
  end
  assert(render_count==lrender_count,'cad models should have the same number of renders')
end

n_fold = render_count
n_images = 2
batch_size = 32
data = torch.FloatTensor(n_fold,n_images,3,224,224)

cc = 0
tt = torch.Timer()
for m_idx,dir in ipairs(image_list) do
  tt:reset()
  --cc = cc + 1
  print(('Processing model %s (%05d/%05d)... '):format(dir,m_idx,#image_list))
  local names = {}
  for imfile in paths.iterfiles(paths.concat(image_path,dir)) do
    table.insert(names,imfile)
  end
  table.sort(names)
  for i,k in ipairs(names) do
    I = image.load(paths.concat(image_path,dir,k))
    if opt.color == 'gray' then
      I = image.rgb2y(I)
    end
    data[i][1] = image.scale(image_transformer:preprocess(I):float(),224,224)
    data[i][2] = image.scale(addRandomBG(I,240/255),224,224) -- 254
  end
  features = compute_fc7(data:view(-1,3,224,224),batch_size)
  local ss = features[1]:size()
  features = features:view(n_fold,n_images,table.unpack(ss:totable()))
  if not opt.debug then
    f:write('/'..dir..'/features',features,options3)
  end
  print(('   Done! Elapsed time: %.2fs'):format(tt:time().real))
  if opt.debug then
    break
  end
end

if not opt.debug then
  f:close()
end
