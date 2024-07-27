%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% inH, inV: factor matrices H, and V at the current iteration
% blockV: the preprocessed block related to the factor matrix V
% blockCAHSk: intermediate data computed with factor matrices A, C, H, Sk
% B: the number of blocks for the given range
% OUTPUT
% outW: updated factor matrix W
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function outW = query_updateW(inH, inV, blockV, blockCAHSk, B)

    VTV = cellfun(@(x) x'*inV, blockV, 'UniformOutput', false);
    
    outW = cell(B,1);
    
    for b=1:B
       outW{b} = cell2mat(cellfun(@(x) x(:)', blockCAHSk{b}, 'UniformOutput', false)) ...
           * khatrirao(VTV{b}, inH)/ ((inV'*inV) .* (inH' * inH));
    end
    
end

