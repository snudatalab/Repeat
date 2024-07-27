%%%%%%%%%%%%%%%%%%%
% This error measurement code refers to the paper below
% Ekta Gujral, Georgios Theocharous and Vagelis Papalexakis - University of
% California,Riverside,CA.Computer Science (2019-2020)
% "SPADE:Streaming PARAFAC2 DEcompistion for Sparse Datasets",
%%%%%%%%%%%%%%%%%%%
function [fit]=pf2fitout(X,Q,H,B,C,K, Z)

  
   fit = zeros(K,1);
   normX = 0;
   for k = 1:K
     M= (Q{k})*H*diag(C(k,:))*B';
     fit(k) = norm(X{k} -M, 'fro')^2;
     normX = normX + norm(X{k} ,'fro')^2;
   end
   
   fit = sum(fit)/normX;
end