function  [E,G] = DAHOG(X,DG,DE,SG,SE,alpha,beta,lambda,k,maxiter)

%%%%%%%%%%%%%%%%%%%%
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples (words) and features (documents), respectively.
%% SE: Similarity matrix in R_+^(m*m) associated with the data samples (words).
%% DE: Degree matrix in R_+^(m*m) obtained from SE.
%% SG: Similarity matrix in R_+^(n*n) associated with the data features (documents).
%% DG: Degree matrix in R_+^(n*n) obtained from SG.
%% alpha, beta, lambda: Regularization parameters.
%% k: The number of topics.
%% m: The numbers of samples (words)
%% n: The numbers of features (documents)
%% maxiter: Maximum number of iterations

XX = (X')*X;
XXX = X * (X');
[m,n] = size(X);

%% Initialization
E = rand(m,k);
G = rand(k,n);

iter=1;
while iter<=maxiter
    %% Update the topic-document matrix G
    XXG = G*XX;
    EX = (E')*X;
    numG = 3*EX + 2*G*(EX')*EX + beta*G*SG + lambda*G;
    denG = (E')*E*G + G + XXG + (XXG*(G')*G + G*(G')*XXG) + alpha*G*DG + lambda*G*(G')*G;
    reG = numG./max(denG,1e-10);
    reG = nthroot(reG,4);
    G = G.*reG;
    
    %% Update the word-topic matrix E
    XG = X*(G');
    XXE = XXX*E;
    numE = 3*X*(G') + 2*XG*(XG')*E + alpha*SE*E;
    denE = E*G*(G') + XXE + E + (XXE*(E')*E + E*(E')*XXE) + alpha*DE*E;
    reE = numE./max(denE,1e-10);
    reE = nthroot(reE,4);
    E = E.*reE;
    %%
    iter = iter+1;
end

end
