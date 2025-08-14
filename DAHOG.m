function  [W,H] = DAHOG(X,XX,XXX,DH,DW,SH,SW,alpha,beta,lambda,k,m,n,maxiter)
%% Initialization
W = rand(m,k);
H = rand(k,n);
iter=1;
while iter<=maxiter
    %% Update H
    XXH = H*XX;
    WX = (W')*X;
    numH = 3*WX + 2*H*(WX')*WX + alpha*H*SH + lambda*H;
    denH = (W')*W*H + H + XXH + (XXH*(H')*H + H*(H')*XXH) + alpha*H*DH + lambda*H*(H')*H;
    reH = numH./max(denH,1e-10);
    reH = nthroot(reH,4);
    H = H.*reH;
    %% Update W
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
