%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% prepQ, prepH, prepW, prepV: preprocessed results 
% R: target rank
% block_size: block size used in the preprocessing phase
% ts and te: start time range and end time range, respectively
% maxiters: maximum number of iterations
% conv: convergence hyperparameter
% OUTPUT
% Q, H, V, W: output factor matrices for the given time range [ts:te]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,H,W,V] = query(prepQ, prepH, prepW, prepV, R, block_size, ts, te, maxiters, conv)
    flag = 0;
    fit = 0;
    oldfit = fit*2;
    S = ceil(ts/block_size);
    E = ceil(te/block_size);
    B = E-S+1;
    
    blockQ = prepQ(S:E);
    blockH = prepH(S:E);
    blockW = prepW(S:E);
    blockV = prepV(S:E);
    
    if rem(ts, block_size) == 0
        blockQ{1} = blockQ{1}(block_size:end);
        blockW{1} = blockW{1}(block_size:end,:);
    else
        blockQ{1} = blockQ{1}(rem(ts, block_size):end);
        blockW{1} = blockW{1}(rem(ts, block_size):end,:);
    end
    
    if rem(te, block_size) == 0
        blockQ{end} = blockQ{end}(1:end);
        blockW{end} = blockW{end}(1:end,:);
    else
        blockQ{end} = blockQ{end}(1:rem(te, block_size));
        blockW{end} = blockW{end}(1:rem(te, block_size),:);
    end 
    
    blockHSV = cell(B,1);
    for b=1:B
        blockHSV{b} = cell(size(blockW{b},1),1);
        for k=1:size(blockW{b},1)
            blockHSV{b}{k} = blockH{b} * diag(blockW{b}(k,:)) * blockV{b}';
        end
    end
    

    Wres = blockW;
    for b=1:size(Wres,1)
        Wres{b} = sort(Wres{b}, 2, 'descend');
        Wres{b} = Wres{b}(:,1:R);
    end
    
    Vres = mean(cat(3, blockV{:}),3);
    Vres = Vres(:,1:R);

    Hres = rand(R,R);

    Ares = cell(B,1);
    Cres = cell(B,1);
    blockHW = cell(B,1);
    for b=1:B
        blockHW{b} = cell(size(blockW{b},1),1);
        for k=1:size(blockW{b},1)
            blockHW{b}{k} = blockH{b}*diag(blockW{b}(k,:));
        end
    end
    
    block_tr = 0;
    for b=1:B
        blockVtV = blockV{b}'*blockV{b};
        for k=1:size(blockW{b},1)
        block_tr = block_tr + trace(blockHW{b}{k}*blockVtV*blockHW{b}{k}');
        end
    end
    for iter=1:maxiters
        oldfit = fit;
        [Ares, Cres, Hres,Wres,Vres] = stitching(Hres, Wres, Vres, blockHW, blockV, B, R);
        fit = block_tr;
        for b=1:B
            VbTV = blockV{b}'*Vres;
            VTV = Vres'*Vres;
            for k=1:size(blockW{b},1)
                ACHSk = Ares{b}{k}*Cres{b}{k}'*Hres*diag(Wres{b}(k,:));
                fit = fit-2*trace(blockHW{b}{k}*VbTV*ACHSk') ...
                    + trace(ACHSk*VTV*ACHSk');
            end    
        end
        
        if ((iter > 1) &&  (abs(fit-oldfit)<oldfit*conv))
            flag = 1;
        else
            flag = 0;
        end

        fprintf(' Iter %2d: fitchange = %7.1e\n', iter, oldfit-fit);
        if flag == 1
            break
        end        
        
    end
    Q = cell(B,1);
    for b=1:B
       Q{b} = cell(size(Ares{b}, 1), 1);
       for k=1:size(Ares{b}, 1)
           Q{b}{k} = blockQ{b}{k} * Ares{b}{k} * Cres{b}{k}';
       end    
    end

    
    Q = cat(1, Q{:});
    H = Hres;
    W = cell2mat(Wres);
    V = Vres;
    
end

