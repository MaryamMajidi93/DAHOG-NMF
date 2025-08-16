function [L,D,S] = makeSD(X,option)
S = constructW(X,option);
D = diag(sum(S));
L = D - S;
end