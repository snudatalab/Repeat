%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% inH, inW: factor matrices H, and W at the current iteration
% blockV: the preprocessed block related to the factor matrix V
% blockCAHSk: intermediate data computed with factor matrices A, C, H, Sk
% B: the number of blocks for the given range
% R: target rank
% OUTPUT
% outW: updated factor matrix W
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function outV = query_updateV(inH, inW, blockV, blockCAHSk, B, R)
    
    outV = zeros(size(blockV{1}, 1), R);

    for b=1:B
       outV = outV + blockV{b}*cell2mat(blockCAHSk{b})' * khatrirao(inW{b}, inH);
    end  

    W_cat = cell2mat(inW);
    outV = outV / ((W_cat' * W_cat) .* (inH' * inH));
    lambda = max( max(abs(outV),[],1), 1 )'; %max-norm
    outV = bsxfun(@rdivide, outV, lambda');     
    
end

