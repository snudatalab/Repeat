%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% inH, inW, inV: factor matrices H, V, and W at the current iteration
% blockHW: the preprocessed block related to HxSk
% blockV: the preprocessed block related to the factor matrix V
% B: the number of blocks for the given range
% OUTPUT
% outA, outC: updated factor matrices A and C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outA, outC] = query_updateQ(inH, inW, inV, blockHW, blockV, B)

    outA = cell(B,1);   
    outC = cell(B,1);
    VTV = cellfun(@(x,y,z) x'*inV, blockV, 'UniformOutput', false);
    for b=1:B
       outA{b} = cell(size(blockHW{b},1),1);
       outC{b} = cell(size(blockHW{b},1),1);
       for k=1:size(blockHW{b},1)
           atmp = blockHW{b}{k}*VTV{b}*diag(inW{b}(k,:))*inH';
           [u,~,v] = svd(atmp, 'econ');
           outA{b}{k} = u;
           outC{b}{k} = v;
       end
    end
end

