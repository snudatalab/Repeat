%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% inH, inW, inV: factor matrices H, V, and W obtained at the previous
% iteration
% blockHW: the preprocessed block related to HxSk
% blockV: the preprocessed block related to the factor matrix V
% B: the number of blocks for the given time range
% R: target rank
% OUTPUT
% outA, outC, outH, outW, outV: factor matrices updated at the current
% iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outA, outC, outH, outW, outV] = stitching(inH, inW, inV, blockHW, blockV, B, R)
    [outA, outC] = query_updateQ(inH, inW, inV, blockHW, blockV, B);
    blockCAHSk = cell(B,1);
    for b=1:B
        blockCAHSk{b} = cell(size(blockHW{b},1), 1);
        for k=1:size(blockHW{b},1)
           blockCAHSk{b}{k} = outC{b}{k}*outA{b}{k}'*blockHW{b}{k};
        end
    end
    outH = query_updateH(inW, inV, blockV, blockCAHSk, B, R);
    outV = query_updateV(outH, inW, blockV, blockCAHSk, B, R);
    outW = query_updateW(outH, outV, blockV, blockCAHSk, B);                    
    

    
end

