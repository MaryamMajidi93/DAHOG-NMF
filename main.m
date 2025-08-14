%% An implementation of Dual Autoencoder-like NMF method with Higher-Order Graph Regularization
%% Maryam Majidi, Farid Saberi Movahed
%% Email: maryam.majidi93@gmail.com & fdsaberi@gmail.com
%% 8-14-2025
%%
%% DAHOG-NMF: Proposed Method 
%%
%% Inputs
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples (words) and features (documents), respectively.
%% S: Similarity matrix associated with the data samples.
%% D: Degree matrix obtained from S.
%% L: Laplacian matrix, L = D - S.
%% alpha, beta, nu: Regularization parameters.
%% r: Vector of size of each layer, r = [r_1, r_2,...,r_l] in which r_i is the size of each layer.
%% l: Number of layers.
%% maxiter: Maximum number of iterations
%%
%% Outputs
%% H: Basis Matrix.
%% V: Coefficient Representation matrix. 
clc
clear
close all
format shortG
addpath('./AdditionalFiles');
