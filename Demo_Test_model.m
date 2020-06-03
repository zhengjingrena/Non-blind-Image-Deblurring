


%%% test the model performance
% clear all;
% reset(gpuDevice(1)) ;
% run(fullfile(fileparts(mfilename('fullpath')), ...
%     '..', '..', 'matlab', 'vl_setupnn.m')) ;
% clear; clc;
format compact;
global ASRmtx  ASRmtx_T


% load(fullfile('data','levin.mat'));
% img_index = 1;
% psf_index = 8; % 1 - 8
% ASRmtx = levin(img_index, psf_index).h; % kernel

PSF = fspecial('average',7);
ASRmtx = PSF; % kernel

DB = 40;


showResult  = 1;
useGPU      = 1;
pauseTime   = 0;


load(fullfile('data','LEARN_Model','LEARN_Model-epoch-40.mat'));

net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);

net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end


% for i = 2
    %     input = single(imread(fullfile('data','test400','input',['00200',num2str(i),'.jpg'])));
    %     inputf = single(inputsf(:,1,i));
%     label = single(imread(fullfile('data','test400','label',['00200',num2str(i),'.jpg'])));
label = im2double(imread('cameraman.tif'));
% figure,imshow(label);
for i =1:3
    label3(:,:,i) = label;
end    
    input = imfilter(label3,ASRmtx);
figure,imshow(input);
    
%                 input = real(imfilter(label,ASRmtx,'symmetric','conv'));
%  input = im2double(uint8(input));

    V_noise = var(label3(:))/10^(DB/10);
    input = imnoise(input,'gaussian',0,V_noise);

    In = single(input);
    figure,imshow(In);

    %%% convert to GPU
    if useGPU
        In = gpuArray(In);fi
    end
    %     tic;
    
    res    = vl_simplenn(net,In,In,[],[],'conserveMemory',true,'mode','normal');
    %     toc
    output = res(end).x;
    
    if useGPU
        output = gather(output);
        input  = gather(input);
    end

[peaksnr,mse] = psnr(label3,output);

