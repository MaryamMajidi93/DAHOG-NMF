%% An implementation of Dual Autoencoder-like NMF method with Higher-Order Graph Regularization
%% Maryam Majidi & Farid Saberi Movahed
%% Emails: maryam.majidi93@gmail.com & fdsaberi@gmail.com
%% 8-14-2025
%%
%% DAHOG-NMF: Proposed Method
%%
%% X: Data set in R_+^(m*n),  where m and n are the numbers of samples (words) and features (documents), respectively.
%% SE: Similarity matrix in R_+^(m*m) associated with the data samples (words).
%% DE: Degree matrix in R_+^(m*m) obtained from SE.
%% SG: Similarity matrix in R_+^(n*n) associated with the data features (documents).
%% DG: Degree matrix in R_+^(n*n) obtained from SG.
%% L: Laplacian matrix, L = D - S.
%% alpha, beta, lambda: Regularization parameters.
%% kmanifold: The k-neighborhood parameter used for constructing the first-order neighbor graph.
%% NTopics: The number of topics.
%% m: The numbers of samples (words)
%% n: The numbers of features (documents)
%% maxiter: Maximum number of iterations


%% Outputs
%% G: Topic-document matrix.
%% E: Word-topic matrix.
clc
clear
close all
format shortG
addpath('./AdditionalFiles');


alpha = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12] %% This parameter needs to be tuned.
lambda = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12]; %% This parameter needs to be tuned.
kmanifold = [5, 10, 15, 20]; %% This parameter needs to be tuned.
NTopics =[3:10, 15, 20, 25]; 
maxiter = 100;

runNMI=zeros(50,1);
runACC=zeros(50,1);
runCoh=zeros(50,1);
runARI=zeros(50,1);

ACCmean  =  zeros(length(alpha),length(lambda),length(kmanifold));
NMImean  =  zeros(length(alpha),length(lambda),length(kmanifold));
Cohmean  =  zeros(length(alpha),length(lambda),length(kmanifold));
ARImean  = zeros(length(alpha),length(lambda),length(kmanifold));

%% Load data
load('TDT2\TDT2.mat'); %% Input data contains fea and gnd.

for NT=NTopics
    for i=1:length(alpha) %% Grid Search for alpha parameter
        for ii=1:length(lambda)  %% Grid Search for lambda parameter
            for iii=1:length(kmanifold) %% Grid Search for kmanifold parameter
                for run = 1:50  %% For 50 Monte Carlo runs
                    SS =[num2str(run) '.mat'];
                    SS = ['TDT2\' num2str(NT) 'Class\' SS];
                    load(SS);
                    fea2 = fea(sampleIdx,:);
                    gnd2 = gnd(sampleIdx,:);

                    fea2(:,zeroIdx) = [];

                    II = unique(gnd2);
                    for kk = 1:length(II)
                        for l = 1:length(gnd2)
                            if gnd2(l)==II(kk)
                                gnd2(l) = kk;
                            end
                        end
                    end

                    X = NormalizeFea(fea2,1); % fea is document-word matrix
                    X = X';
                    %% Please not that you should put the data samples in the columns of data and the final X will be word-document matrix
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    [DG,SG] = makeLaplacian(X', kmanifold,alpha,lambda);
                    [DE,SE] = makeLaplacian(X, kmanifold,alpha,lambda);
                    [E,G] = DAHOG(X,DG,DW,SG,SW,alpha,alpha,lambda,NT,maxiter);

                    c =  length(unique(gnd2));
                    %% Clustering
                    tempNMI=zeros(20,1);
                    tempACC=zeros(20,1);
                    tempCoh=zeros(20,1);
                    tempARI=zeros(20,1);
                    for j = 1:20
                        IDX = kmeans(G',c);
                        tempNMI(j) = nmi(gnd2,IDX);
                        tempACC(j) = clusterAccMea(gnd2,IDX);
                        tempCoh(j) = coherence(E,X,25);
                        tempARI(j) = randindex(gnd2, IDX);
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    runACC(run) = mean(tempACC);
                    runNMI(run) = mean(tempNMI);
                    runCoh(run) = mean(tempCoh);
                    runARI(run) = mean(tempARI);
                end


                ACCmean(i,ii,iii) = mean(runACC);
                NMImean(i,ii,iii) = mean(runNMI);
                Cohmean(i,ii,iii) = mean(runCoh);
                ARImean(i,ii,iii) = mean(runARI);
            end
        end
    end
end
