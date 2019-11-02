function [ap,prec,rec,cls,imlist] = evaluate_detection(detect_file,t_aspect_r)
cad_set = 'cvprchairs';%'mathieuCVPR2014_normals'; % cvprchairs, ikeaobject, francisco_normals_chair
test_set = 'voc2012subset'; %'voc2012subset', 'ikeaobject'
ext = 'png';
use_aspect_ratio = true;
nms_value = 0.3;

paths = detectInit();

detect_info = h5info(detect_file);

imlist = getSynthImageNames(paths.cadDataset.(cad_set).folder,ext);

% load ss bboxes
dt = getSSBoxes(test_set);

aspect_ratios = getAspectRatios(cad_set,imlist);
groundtruth = getGroundTruth(test_set);

%

cls = fieldnames(groundtruth.classes);
if strcmp(cad_set,'ikeaobject'),
  % hack for the moment, as the orders are inverted
  % as there is no malm1 in Lim's data, I'm duplicating
  % malm2, but that should be fixed (tomorrow)
  % this reordering correspond to the alphabetical order
  % in the original folder names of the cad models.
  %cls = cls([10,2,1,7,8,5,1,4,3,6,9]);
  cls = cls([7,9,6,8,2,3,4,5,10,1,1]);
end
num_classes = length(cls);

detections = cell(1,num_classes);
ap = zeros(1,num_classes);
prec = cell(1,num_classes);
rec = cell(1,num_classes);

num_images = length(dt.images);

for cl_idx=1:num_classes, %classes
    %disp(cl_idx)
    
    % hack for the moment, improve tomorrow
    if strcmp(cad_set,'ikeaobject') == 1,
      clsElms = false(1,numel(clsElms));
      clsElms(216*(cl_idx-1)+(1:216)) = 1;
    else
      clsElms = paths.cadDataset.(cad_set).getClassElms(imlist,cls{cl_idx});
    end

    clsElmsIdx = find(clsElms);
    
    dets = cell(1,num_images);

    for i = 1:num_images,
        img_info = detect_info.Groups(i);
        %img_name = img_info.Name(2:end);
        
        %assert(strcmp(img_name,dt.images{i}))

        %bb_dist     = h5read(detect_file,[img_info.Name,'/distances'])';
        img_name    = dt.images{i};
        bb_dist     = h5read(detect_file,['/',img_name,'/distances'])';
        
        if length(img_info.Datasets) > 1,
            indexes = h5read(detect_file,['/',img_name,'/indexes'])';
            indexes = double(indexes);
        else
            indexes = repmat(1:size(bb_dist,2),[size(bb_dist,1),1]);
        end
        
        % format: [Ymin Xmin Ymax Xmax]
        boxes = dt.boxes{i};
        
        aspect_ratios_here = aspect_ratios(indexes);
        
        if use_aspect_ratio,
            l_aspect_r = (boxes(:,4)-boxes(:,2))./(boxes(:,3)-boxes(:,1));
            r_asp_r = bsxfun(@rdivide,l_aspect_r,aspect_ratios_here);
            r_asp_r = r_asp_r > 1/t_aspect_r | r_asp_r < t_aspect_r;
            
            bb_dist(r_asp_r) = -inf;
        end
        
        % set cad models scores from other classes to -inf
        % need to take care of indexes as well
        bb_dist(~ismember(indexes,clsElmsIdx)) = -inf;
        
        % take maximum over each bbox taking into account the specific class
        [confidence,~] = max(bb_dist,[],2);
        remove_inf = isinf(confidence);
        confidence(remove_inf) = [];
        boxes(remove_inf,:) = [];
        
        assert(any(any(ismember(indexes,clsElmsIdx))))
        
        % do nms
        nms_idx = nms([boxes,confidence],nms_value);
        boxes = boxes(nms_idx,:);
        confidence = confidence(nms_idx);
        
        [confidence,idx] = sort(confidence,'descend');
        boxes = boxes(idx,:);
         
        confidence = double(confidence);
        
        dets{i} = [boxes confidence i*ones(length(confidence),1)];
        
    end
    detections{cl_idx} = cat(1,dets{:});
    
    [rec{cl_idx},prec{cl_idx},ap(cl_idx)] = VOCevaldet_here(...
        groundtruth.classes.(cls{cl_idx}), detections{cl_idx}, test_set);
end


end

function dt = getSSBoxes(test_set)
paths = detectInit();
dt = load(paths.testDataset.(test_set).ssBoxesFile);
end

function groundtruth = getGroundTruth(test_set)
groundtruth = load('groundtruths.mat');
groundtruth = groundtruth.(test_set);
end

function imlist = getSynthImageNames(cadFolder,ext)
% load synthetic images
d = dir(cadFolder);
isub = [d(:).isdir]; %# returns logical vector
imlist_base = {d(isub).name}';
imlist_base(ismember(imlist_base,{'.','..'})) = [];
imlist_base = sort(imlist_base);

if false
fid = fopen('/home/francisco/work/projects/cross_domain/torch_new/mathieu_chairs/mathieu_chairs_idx_list.txt');
imlist_base = {};
tline = fgetl(fid);
while ischar(tline)
    imlist_base{end+1} = tline;
    tline = fgetl(fid);
end
fclose(fid);
end

imlist = {};
count = 0;
for i=1:length(imlist_base),
    cadName = imlist_base{i};
    renderName = dir(fullfile(cadFolder,cadName,['*.',ext]));
    renderName = {renderName.name};
    renderName = sort(renderName);
    for j=1:length(renderName),
        count = count + 1;
        imlist{count} = [cadName,'/',renderName{j}];
    end
end
end

function aspect_ratios = getAspectRatios(cad_set,imlist)
paths = detectInit();
if true,%exist([cad_set,'_aspect_ratio.mat'],'file') == 0,
    disp('Computing aspect ratios')
    aspect_ratios = zeros(1,length(imlist));
    for i=1:length(imlist),
        iminfo = imfinfo(fullfile(paths.cadDataset.(cad_set).folder,imlist{i}));
        aspect_ratios(i) = iminfo.Width/iminfo.Height;
    end
    %save([cad_set,'_aspect_ratio.mat'],'aspect_ratios')
else
    aspect_ratios = load([cad_set,'_aspect_ratio.mat']);
    aspect_ratios = aspect_ratios.aspect_ratios;
end
end

% copy from VOCdevkit with slight modifications with respect to data
% loading
function [rec,prec,ap] = VOCevaldet_here(groundtruth, detections, dataset)
minoverlap = 0.5;

% extract ground truth objects

% number of images
M = max(max(detections(:,end)),max(groundtruth(:,end)));

npos=0;
% GT format: [Ymin Xmin Ymax Xmax Image# Diff Occ Trunc]
gt(M)=struct('BB',[],'diff',[],'det',[]);
gt_idx = unique(groundtruth(:,5));
for i=1:length(gt_idx)
    % extract objects of class
    im_idx = gt_idx(i);
    clsinds = find(groundtruth(:,5) == im_idx);
    gt(im_idx).BB=groundtruth(clsinds,1:4)';
    gt(im_idx).diff=any(groundtruth(clsinds,6:end),2)';
    gt(im_idx).det=false(length(clsinds),1);
    npos=npos+sum(~gt(im_idx).diff);
end

% load results
BB = detections(:,1:4)';
ids = detections(:,end);
confidence = detections(:,5);

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);

for d=1:nd    
    % find ground truth image
    i=ids(d);
    
    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
                gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

if strcmp(dataset,'ikeaobject'),
    ap=LIMap(rec,prec);
else
    ap=VOCap(rec,prec);
end

end

function ap = LIMap(rec,prec)
ap = sum((prec(2:end)+prec(1:end-1))/2.*(rec(2:end)-rec(1:end-1)));
end

function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end
