%% mathieu chairs

addpath('/home/francisco/work/datasets/VOCdevkit/VOCcode')
VOCinit;
dtgt = load(fullfile('../torch_new/mathieu_chairs/mathieu_chairs_gt_bboxes.mat'));

gt = struct();
dd = {};

for model_idx = 1:length(dtgt.images)
    
    recs=PASreadrecord(sprintf(VOCopts.annopath,dtgt.images{model_idx}));
    clsinds=strmatch('chair',{recs.objects(:).class},'exact');
    
    bbs = dtgt.boxes{model_idx};
    
    for gt_id = 1:size(bbs,1)
        oobj = recs.objects(clsinds(gt_id));
        %info0.objects(gt_id).class = 'chair';
        %info0.objects(gt_id).difficult = oobj.truncated | oobj.occluded | oobj.difficult;
        %info0.objects(gt_id).bbox = bbs(gt_id,:);%[2,1,4,3]
        dd{end+1} = [bbs(gt_id,:) model_idx oobj.difficult oobj.occluded oobj.truncated];
    end
    
    %info00{model_idx} = info0;
end
imlist = dtgt.images;
DD = cat(1,dd{:});
gt.classes = struct();
gt.classes.chair = DD;
gt.imList = imlist;
clear info0 info00
save('../mathieu_chairs_gt.mat','gt');

%% voc 2007
cls = {'aeroplane',...
    'bicycle',...
    'bird',...
    'boat',...
    'bottle',...
    'bus',...
    'car',...
    'cat',...
    'chair',...
    'cow',...
    'diningtable',...
    'dog',...
    'horse',...
    'motorbike',...
    'person',...
    'pottedplant',...
    'sheep',...
    'sofa',...
    'train',...
    'tvmonitor'};

curr_dir = pwd;
vocdir = '/media/francisco/45c5ffa3-52da-4a13-bdf4-e315366c2bdb/francisco/datasets/VOCdevkit';
vocdevkitdir = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/VOCdevkit/';
cd(vocdir);
addpath(fullfile(vocdevkitdir,'VOCcode'))
VOCinit;
cd(curr_dir);
gtids=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s');
gt = struct();
dd = struct();
for i=1:length(cls),
    gt.(cls{i}) = struct('data',[],'imList',{});
    dd.(cls{i}) = {};
end


imlist = {};
for i=1:length(gtids),
    recs=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    %clsinds=strmatch(cls{cc},{recs.objects(:).class},'exact');
    clsinds = 1:size(recs.objects,2);
    for gt_id = 1:length(clsinds)
        oobj = recs.objects(clsinds(gt_id));
        %info0.objects(gt_id).class = oobj.class;%'chair';
        %info0.objects(gt_id).difficult = oobj.difficult; %oobj.truncated | oobj.occluded |
        %info0.objects(gt_id).bbox = oobj.bbox([2,1,4,3]);% y1, x1, y2, x2
        dd.(oobj.class){end+1} = [oobj.bbox([2,1,4,3]) i oobj.difficult 0 0];
        
    end
    
    imlist{end+1} = gtids{i};
    
    
end

gt = struct();
gt.classes = struct();
for i=1:length(cls),
    gt.classes.(cls{i}) = cat(1,dd.(cls{i}){:});
end
gt.imList = imlist;

save('../voc2007_gt.mat','gt');

%% ikea
datafolder = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/';
rfold = '/media/francisco/7b0c9380-7b14-4999-a0d4-a91d079a151c/datasets/IKEA3D/lim_results';
flist = dir(fullfile(rfold,'*.txt'));
flist = {flist.name};

% read image list
imlist = {};
fid = fopen(fullfile(rfold,'image_list.txt'));
tline = fgetl(fid);
while ischar(tline)
    imlist{end+1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

% get data
gt_files = flist(cellfun(@(x) ~isempty(strfind(x,'gt')), flist));
cls = cellfun(@(x) x(1:end-7), gt_files, 'UniformOutput',false);

num_classes = length(gt_files);
gt = struct();
gt.classes = struct();
% do the evaluation
for idx=1:num_classes,
    % GT format: [Xmin Ymin Xmax Ymax Image#]
    groundtruth = dlmread(fullfile(rfold,gt_files{idx}));
    
    bndbox = groundtruth(:,1:4);
    bndbox = bndbox(:,[2,1,4,3]);
    
    for j=1:size(bndbox,1),
        iminfo = imfinfo(fullfile(datafolder,imlist{groundtruth(j,end)}));
        % lim consider scale == 1, I considered it as iminfo.Width/500
        scale = iminfo.Width/500;
        %scale = 1;

        bndbox(j,:) = (bndbox(j,:)-1)*scale + 1;
        
    end
    % convert to [Ymin Xmin Ymax Xmax Image# Diff Occ Trunc]
    gt.classes.(cls{idx}) = [bndbox groundtruth(:,5),zeros(size(groundtruth,1),3)];
    
end
gt.imList = imlist;

save('../ikeaobject_gt.mat','gt');

%% get them all
clear all; clc;
voc2012subset = load('../mathieu_chairs_gt.mat');
voc2012subset = voc2012subset.gt;
voc2007 = load('../voc2007_gt.mat');
voc2007 = voc2007.gt;
ikeaobject = load('../ikeaobject_gt.mat');
ikeaobject = ikeaobject.gt;

save('../groundtruths.mat','voc2012subset','voc2007','ikeaobject');