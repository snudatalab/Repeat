%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% X: an input irregular tensor
% R: target rank
% block_size: block size used in the preprocessing phase
% total_length: the number of slice matrices (equal to the total time
% length
% OUTPUT
% prepQ, prepH, prepW, prepV: preprocessed blocks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [prepQ, prepH, prepW, prepV] = preprocessing(X, R, block_size, total_length)

    B = ceil(total_length/block_size);
    K = size(X,1);

    prepQ = cell(B,1);
    prepH = cell(B,1);
    prepW = cell(B,1);
    prepV = cell(B,1);
    
    for b=1:B
        Maxiters = 20;
        Constraints = [0 0];
        Options = [1e-6 Maxiters 2 0 0 0 0];

        block_st = (b-1)*block_size+1;
        if b*block_size > size(X,1)
            block_end = size(X,1);
        else
            block_end = b*block_size;
        end
        
        
        Y = X(block_st:block_end);        
        [H, V, W, Q, ~, ~]=parafac2_sparse_paper_version(Y,R,Constraints,Options);
        
        prepQ{b} = Q;
        prepH{b} = H;
        prepW{b} = W;
        prepV{b} = V;
        
    end    
    

end

