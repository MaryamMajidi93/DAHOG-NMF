%% An implementation of Dual Autoencoder-like NMF method with Higher-Order Graph Regularization
%% Maryam Majidi & Farid Saberi Movahed
%% Emails: maryam.majidi93@gmail.com & fdsaberi@gmail.com
%% 8-14-2025
%%
%% DAHOG-NMF: Proposed Method 
%%
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples (words) and features (documents), respectively.
%% SW: Similarity matrix in R_+^(m*m) associated with the data samples (words).
%% DW: Degree matrix in R_+^(m*m) obtained from SW.
%% SH: Similarity matrix in R_+^(n*n) associated with the data features (documents).
%% DH: Degree matrix in R_+^(n*n) obtained from SH.
%% L: Laplacian matrix, L = D - S.
%% alpha, beta, lambda: Regularization parameters.
%% kmanifold: The k-neighborhood parameter used for constructing the first-order neighbor graph.
%% k: The number of topics.
%% m: The numbers of samples (words)
%% n: The numbers of features (documents)
%% maxiter: Maximum number of iterations


%% Outputs
%% V: Topic-document matrix.
%% U: Word-topic matrix. 
clc
clear
close all
format shortG
addpath('./AdditionalFiles');


alpha = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12] %% This parameter needs to be tuned.
lambda = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12]; %% This parameter needs to be tuned.
kmanifold = [5, 10, 15, 20]; %% This parameter needs to be tuned.
k =[3:10, 15, 20, 25]; %% The number of topics.
maxiter = 100;

%% load data contaning fea and gnd
X = NormalizeFea(fea,1); % fea is document-word matrix
X = X';
%% Please not that you should put the data samples in the columns of data and the final X will be word-document matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[DH,SH] = makeLaplacian(X', kmanifold,alpha,lambda);
[DW,SW] = makeLaplacian(X, kmanifold,alpha,lambda);
[U,V] = OurDualAuto(X,DH,DW,SH,SW,alpha,alpha,lambda,k,maxiter);
