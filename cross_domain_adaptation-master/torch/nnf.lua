require 'nn'
require 'image'
--require 'inn'
require 'xlua'

nnf = {}

torch.include('nnf','ImageTransformer.lua')

--torch.include('nnf','DataSetPascal.lua')
--torch.include('nnf','BatchProvider.lua')

--torch.include('nnf','SPP.lua')
torch.include('nnf','RCNN.lua')

--torch.include('nnf','Trainer.lua')
--torch.include('nnf','Tester.lua')

--torch.include('nnf','SVMTrainer.lua')



--torch.include('nnf','ContrastiveEmbeddingCriterion.lua')

torch.include('nnf','L2Normalize.lua')
--return nnf
