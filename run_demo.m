function [all_T,all_img] = run_demo(strDir,opts)
%% input data
patch_frames = opts.patch_frames;
imgDir = dir([strDir  '*.jpg']);
len = length(imgDir);
for i=1:patch_frames:len-patch_frames+1
    tic
    DD = [];
    for j = i:patch_frames+i-1
        img = imread([strDir imgDir(j).name]);
        [m, n, ch]=size(img);      
        if ch==3
            img=rgb2gray(img);
        end
        DD = cat(3,DD,img);
    end
    DD = double(DD);
    Xsum = sum(DD,3);
    Xsum = Xsum/max(max(Xsum));
    MM = std(DD,0,3);
    MM = MM/(max(max(MM)));
    NN = 1-MM;
    RR = Xsum.*NN.*NN;
    WW = RR.*DD;%Wcs
    weight = [];
    
   for k=1:patch_frames
        [lambda_1, lambda_2] = structure_tensor_lambda(WW(:,:,k), 3);
        cornerStrength_0 = (((lambda_1.*lambda_2)./(lambda_1 + lambda_2+0.001)));
        maxValue = (max(lambda_1,lambda_2)); 
        cornerStrength = cornerStrength_0.*maxValue;
        priorWeight = mat2gray(cornerStrength);
        weight=cat(3,weight,priorWeight);
    end

    [tenB, tenT,tenN] = run_Func(DD,weight,opts);
    toc
    tenT = gather(tenT);%target results tensor
    %% gather all image and bw
    all_T(:,:,i:i+patch_frames-1) = tenT;
    all_img(:,:,i:i+patch_frames-1) = DD;
end
