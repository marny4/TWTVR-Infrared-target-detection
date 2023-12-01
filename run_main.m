% This matlab code implements "Infrared Maritime Target Detection Based on Temporal Weight
% and Total Variation Regularization Under Strong Wave Interferences"
%
% Written by Enzhong Zhao 
% 2023-Dec-1
clc
clear;
close all;

addpath data/;

%size of input images
m = 512;
n = 640;

%adjust for better results
paraH = 0.22;%
paraL2 = 0.1;
paraL3 = 1;
paraL4 = 0.01;
paramu = 0.003;
parapatch = 10;

Dir = ['./data/dataset01/'];

[sizedir,~] = size(Dir) ;

for dir_i = 1:sizedir
    strDir = Dir(dir_i,:);
    patch_frames = 10;
    H = 0.22;
    opts.lambda1= H / sqrt((max(m,n)*patch_frames));
    opts.lambda2 = 0.1;
    opts.lambda3 = 1;
    opts.lambda4 = 0.01;
    opts.mu = 0.003;
    opts.patch_frames = patch_frames;
    
    
    [all_T, all_img] = run_demo(strDir,opts);
    [~,~,sizeT] = size(all_T);
    for ii = 1:sizeT
        tarImg(:,:,ii) = all_T(:,:,ii)/max(max(all_T(:,:,ii)));%results
        oriImg(:,:,ii) = all_img(:,:,ii)/max(max(all_img(:,:,ii)));
    end
end