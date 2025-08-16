function [D,S] = makeLaplacian(X,kmanifold,alpha,lambda)
%%%   Constrcution of the affinity matrix S
options.NeighborMode = 'KNN';
options.k = kmanifold;
options.WeightMode = 'Binary';
[~,D1,W1] = makeSD(X,options);
W2 = (W1')*W1;
D2 = diag(sum(W2));
D = alpha*D1 + lambda*D2;
S = alpha*W1 + lambda*W2;
end