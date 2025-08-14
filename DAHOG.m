function  [W,H] = DAHOG(X,XX,XXX,DH,DW,SH,SW,alpha,beta,lambda,k,m,n,maxiter)

%%%%%%%%%%%%%%%%%%%%
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples (words) and features (documents), respectively.
%% SW: Similarity matrix in R_+^(m*m) associated with the data samples (words).
%% DW: Degree matrix in R_+^(m*m) obtained from SW.
%% SH: Similarity matrix in R_+^(n*n) associated with the data features (documents).
%% DH: Degree matrix in R_+^(n*n) obtained from SH.
%% alpha, beta, lambda: Regularization parameters.
%% k: The number of topics.
%% m: The numbers of samples (words)
%% n: The numbers of features (documents)
%% maxiter: Maximum number of iterations



%% Initialization
W = rand(m,k);
H = rand(k,n);
iter=1;
while iter<=maxiter
    %% Update the topic-document matrix H
    XXH = H*XX;
    WX = (W')*X;
    numH = 3*WX + 2*H*(WX')*WX + alpha*H*SH + lambda*H;
    denH = (W')*W*H + H + XXH + (XXH*(H')*H + H*(H')*XXH) + alpha*H*DH + lambda*H*(H')*H;
    reH = numH./max(denH,1e-10);
    reH = nthroot(reH,4);
    H = H.*reH;
    
    %% Update the word-topic matrix W
    XH = X*(H');
    XXW = XXX*W;
    numW = 3*X*(H') + 2*XH*(XH')*W + beta*SW*W;
    denW = W*H*(H') + XXW + W + (XXW*(W')*W + W*(W')*XXW) + beta*DW*W;
    reW = numW./max(denW,1e-10);
    reW = nthroot(reW,4);
    W = W.*reW;
    %%
    iter = iter+1;
end

end
