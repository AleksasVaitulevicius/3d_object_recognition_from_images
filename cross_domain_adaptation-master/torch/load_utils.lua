local rootfolder = os.getenv('CACHEFOLDERPATH')

local function loadFeatsName()
  local fname
  if is_old then
    local cachefold = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/projects/cross_domain/cachedir/calibration_features'
    if opt.color == 'rgb' then
      fname = paths.concat(cachefold,'mathieu_chairs.h5')
    elseif opt.color == 'gray' then
      fname = paths.concat(cachefold,'mathieu_chairs_gray_grayforeground.h5')
    else
      -- buggy mixture of color foreground and gray background
      fname = paths.concat(cachefold,'mathieu_chairs_gray.h5')
    end
  else
    local expfolder = 'features'
    local savefolder = paths.concat(rootfolder,opt.name, expfolder)

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Options:')
    cmd:option('-name',opt.name,'')
    cmd:option('-color','rgb','')
    cmd:option('-model','caffenet','')
    cmd:option('-layer','conv5','')
    cmd:option('-debug',false,'')
    cmd:option('-gpu',1,'')

    local temp_opt = cmd:parse(opt)

    fname = cmd:string(temp_opt.name, opt, {gpu=true, name=true,debug=true,
                                            color=false,layer=false,model=false,
                                            })
    fname = paths.concat(savefolder,fname,'features.h5')
    assert(paths.filep(fname),'Features file could not be found '.. fname)
  end
  return fname
end

local function loadProjName()
  local name = opt.projname
  local expfolder = 'projection'
  local savefolder = paths.concat(rootfolder,name, expfolder)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-name',name,'')
  cmd:option('-projname',name,'')
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

  cmd:option('-crandom',false,'completely random patches sampled')
  cmd:option('-calibrate',true,'')
  cmd:option('-debug',false,'')
  cmd:option('-gpu',1,'')
-- [[
  cmd:option('-best_k',opt.best_k or 100,'')
  cmd:option('-batchcomp',opt.batchcomp == nil and true or opt.batchcomp,'')
  cmd:option('-subset',opt.subset or 0,'')
  cmd:option('-subset_type',opt.subset_type or 'rand','')
  --]]
  cmd:option('-lr2',0.01,'lr classif')
  cmd:option('-wd2',5e-4,'wd classif')
  cmd:option('-lr2_step',40,'lr classif step')
  cmd:option('-lr2_div',10,'lr classif division factor')

  cmd:option('-nThreads',6,'')
  cmd:option('-iter_per_thread',8,'')
  cmd:option('-dataset','voc2012subset','dataset to do the evaluation')

  local opt_temp = cmd:parse(arg or {})

  local fname = cmd:string(opt_temp.projname, opt_temp, {gpu=true, name=true,debug=true,
                                          best_k=true, batchcomp=true, subset=true,  subset_type=true,
                                          color=false,layer=false,model=false,
                                          dist=false,crandom=true,calibrate=true,
                                          lr2=true,wd2=true,nThreads=true,iter_per_thread=true,
                                          dataset=true,lr2_step=true,lr2_div=true,
                                        })
  local ext = opt.conv_proj == '' and '.mat' or '.t7'
  fname = paths.concat(savefolder,fname,'projection'..ext)

  assert(paths.filep(fname),'Projection file could not be found ' .. fname)
  return fname
end

local function loadCalibName()
  local expfolder = 'calibration'
  local savefolder = paths.concat(rootfolder,opt.name, expfolder)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-name','mathieu_chairs','')
  cmd:option('-projname',opt.name,'')
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

  cmd:option('-best_k',opt.best_k,'')
  cmd:option('-batchcomp',opt.batchcomp,'')
  cmd:option('-subset',opt.subset,'')
  cmd:option('-subset_type',opt.subset_type,'')

  cmd:option('-dataset','voc2012subset','dataset to do the evaluation')

  local opt_temp = cmd:parse(arg or {})

  local fname = cmd:string(opt_temp.name, opt, {gpu=true, name=true,debug=true,
                                          best_k=true, batchcomp=true, subset=true,
                                          subset_type=true,dataset=true,
                                          color=false,layer=false,model=false,
                                          dist=false,normalize=false,
  --                                        projname=false,
                                        })
  fname = paths.concat(savefolder,fname,'calibration.h5')
  assert(paths.filep(fname),'Calibration file could not be found ' .. fname)
  return fname
end


local function getLinProj()
  local fname = loadProjName()
  print(('Load projection from %s'):format(fname))
  local tmodel
  if opt.conv_proj == '' then
  local d = matio.load(fname,'P')
  if not d then
    d = matio.load(fname,'M_save')
  end
  if type(d) == 'table' then
    d = d[1]
  end
  local w = d[{{1,-2},{1,-2}}]
  local b = d[{{1,-2},-1,}]
  local n_in = w:size(1)
  local lin = nn.Linear(n_in,n_in)
  lin.weight:copy(w)
  lin.bias:copy(b)
  tmodel = nn.Sequential()
  tmodel:add(lin)
 
  if opt.bn then
    local bnmodel = nn.BatchNormalization(n_in)
    local datasave = matio.load(fname,{'bnw','bnb','bnm','bns'})
    bnmodel.weight:copy(datasave.bnw)
    bnmodel.bias:copy(datasave.bnb)
    bnmodel.running_mean:copy(datasave.bnm)
    bnmodel.running_std:copy(datasave.bns)

    tmodel:add(bnmodel)
  end

  if opt.relu then
    tmodel:add(nn.ReLU())
  end
  else
    tmodel = torch.load(fname)
  end
  print('Loaded projection module:')
  print(tmodel)
  return tmodel
end


return loadFeatsName, loadProjName, loadCalibName, getLinProj
