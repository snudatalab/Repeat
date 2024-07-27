%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% inV, inW: factor matrices V and W at the current iteration
% blockV: the preprocessed block related to the factor matrix V
% blockCAHSk: intermediate data computed with factor matrices A, C, H, Sk
% B: the number of blocks for the given range
% R: target rank
% OUTPUT
% outH: updated factor matrix H
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outH] = query_updateH(inW, inV, blockV, blockCAHSk, B, R)

    VTV = cellfun(@(x) x'*inV, blockV, 'UniformOutput', false);
    Htmp = zeros(R,R);
    for b=1:B
        Htmp = Htmp + cell2mat(blockCAHSk{b}') * khatrirao(inW{b}, VTV{b});
%         for k=1:size(blockCAHSk{b}, 1)
%             Htmp = Htmp + blockCAHSk{b}{k} * khatrirao(inW{b}(k,:), VTV{b});
%         end
    end
    W_cat = cell2mat(inW);
    outH = Htmp / ((W_cat' * W_cat) .* (inV' * inV));
% 	if iter == 1
    lambda = max( max(abs(outH),[],1), 1 )'; %max-norm
% 	else
%     lambda = max( max(abs(outH),[],1), 1 )'; %max-norm
% 	end
    outH= bsxfun(@rdivide, outH, lambda');    
end

