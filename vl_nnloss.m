function Y = vl_nnloss(X,c,dzdy,varargin)

% --------------------------------------------------------------------
% pixel-level L2 loss
% --------------------------------------------------------------------
global  ASRmtx
[I_x,I_y]=size(ASRmtx);
if nargin <= 2 || isempty(dzdy)
    t = ((X(round(I_x/2):end-round(I_x/2)+1,round(I_x/2):end-round(I_x/2)+1,:,:)-c(round(I_x/2):end-round(I_x/2)+1,round(I_x/2):end-round(I_x/2)+1,:,:)).^2);
    Y = sum(t(:))/size(X,4); % reconstruction error per sample;
else
    Y = bsxfun(@minus,X,c).*dzdy/size(X,4);
    Y= Y*2;
end

