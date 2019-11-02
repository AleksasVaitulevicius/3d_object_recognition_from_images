require 'nnf'
require 'cunn'
require 'loadcaffe'

local prototxtfile = assert(os.getenv('CAFFENETPROTOTXT'),'Need to provide the path to prototxt')
if not paths.filep(prototxtfile) then
  error('Prototxt does not exist: '..prototxtfile)
end
local modelfile = assert(os.getenv('CAFFENETCAFFEMODEL'),'Need to provide the path to caffemodel')
if not paths.filep(modelfile) then
  error('CaffeModel does not exist: '..modelfile)
end

local usecudnn = os.getenv('USECUDNN')
local backendmodule
local backend
if usecudnn == '1' then
  backend = 'cudnn'
  require 'cudnn'
  backendmodule = cudnn
else
  backend = 'nn'
  backendmodule = nn
end

local function load_models()
  local model_test
  local image_transformer
  local feat_dim
  if opt.model == 'caffenet' then
    print('Loading model '..opt.model)
    local base_model = loadcaffe.load(
                  prototxtfile,
                  modelfile,
                  backend)
    local layerlist = {conv3=10,conv4=12,conv5=16,fc6=18,fc7=21}
    model_test = nn.Sequential()
    for i=1,layerlist[opt.layer] do
      if not (opt.fc7 and i == 19) then
        model_test:add(base_model:get(i):clone())
      end
    end
    if opt.layer == 'conv3' or opt.layer == 'conv4' then
      model_test:add(backendmodule.SpatialMaxPooling(3,3,2,2))
      model_test:add(nn.View(-1):setNumInputDims(3))
    end
    model_test:cuda()
    model_test:evaluate()
    print(model_test)

    if opt.layer == 'conv3' then
      feat_dim = 384*6*6
    elseif opt.layer == 'conv4' then
      feat_dim = 384*6*6
    elseif opt.layer == 'conv5' then
      feat_dim = 256*6*6
    elseif opt.layer == 'fc6' then
      feat_dim = 4096
    elseif opt.layer == 'fc7' then
      feat_dim = 4096
    end

    image_transformer = nnf.ImageTransformer{mean_pix={103.939, 116.779, 123.68},
                                           raw_scale = 255,
                                           swap = {3,2,1}}

  end
  return model_test,image_transformer,feat_dim
end

return load_models
