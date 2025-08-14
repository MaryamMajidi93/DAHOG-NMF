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
%% H: Topic-Document Matrix.
%% W: Word-topic matrix. 
clc
clear
close all
format shortG
addpath('./AdditionalFiles');

runNMI=zeros(50,1);
runACC=zeros(50,1);
runCoh=zeros(50,1);
runARI=zeros(50,1);

alpha = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12];

lambda = [1e-08, 1e-06, 1e-04, 1e-02, 1, 1e02, 1e04, 1e06, 1e08, 1e10, 1e12];

kmanifold = [5, 10]; 


NTopics =[3:10, 15, 20, 25];

ACCmean  =  zeros(length(alpha),length(lambda),length(kmanifold));
NMImean  =  zeros(length(alpha),length(lambda),length(kmanifold));
Cohmean  =  zeros(length(alpha),length(lambda),length(kmanifold));
ARImean  = zeros(length(alpha),length(lambda),length(kmanifold));


%% Load Dataset
%load('/home/392030/TopicNMF/OurDualAutoTDT2/TDT2/TDT2.mat')
load('TDT2\TDT2.mat')

for NT=NTopics
    for i=1:length(alpha) %% Grid Search for alpha
        for ii=1:length(lambda)  %% Grid Search for lambda
            for iii=1:length(kmanifold) %% Grid Search for kmanifold
                for run = 1:50
                    disp(run)
                    SS =[num2str(run) '.mat'];
                    %SS = ['/home/392030/TopicNMF/OurDualAutoTDT2/TDT2/' num2str(NT) 'Class/' SS];
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

                    X = NormalizeFea(fea2,1); % fea2 id doc-word matrix
                    X = X'; % final X will be word-doc matrix
                    %% Please not that you should put the data samples in the columns of data
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% Parameters
                    XX = (X')*X;
                    XXX = X * (X');
                    [m,n] = size(X);
                    maxiter = 200;
                    c =  length(unique(gnd2));
                    k = NT;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% OurDualAuto method

                    [DH,SH] = makeLaplacian(X', kmanifold(iii),alpha(i),lambda(ii));
                    [DW,SW] = makeLaplacian(X, kmanifold(iii),alpha(i),lambda(ii));
                    [U,V] = OurDualAuto(X,XX,XXX,DH,DW,SH,SW,alpha(i),alpha(i),lambda(ii),k,m,n,maxiter);
                    %% Clustering
                    tempNMI=zeros(20,1);
                    tempACC=zeros(20,1);
                    tempCoh=zeros(20,1);
                    tempARI=zeros(20,1);
                    for j = 1:20
                        IDX = kmeans(V',c); %%kmeans  V is topic-doc matrix
                        tempNMI(j) = nmi(gnd2,IDX);   %% NMI
                        tempACC(j) = clusterAccMea(gnd2,IDX); %% ACC
                        tempCoh(j) = coherence_3(U,X,25); %% Coh   (U is word-topic matrix)
                        tempARI(j) = randindex(gnd2, IDX); %% ARI
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    runACC(run) = mean(tempACC);
                    runNMI(run) = mean(tempNMI);
                    runCoh(run) = mean(tempCoh);
                    runARI(run) = mean(tempARI);
                    disp(runARI(run));
                    
                end
                ACCmean(i,ii,iii) = mean(runACC);
                NMImean(i,ii,iii) = mean(runNMI);
                Cohmean(i,ii,iii) = mean(runCoh);
                ARImean(i,ii,iii) = mean(runARI);
                
                meanMats=[ACCmean;NMImean;Cohmean;ARImean];
                Mats =['save meanMats_' num2str(NT) '_Topic   meanMats;'];
                eval(Mats);
                
                fprintf(1,'# of Topics = %d ,mean for alpha%d & lambda%d & kmanifold%d \n',NT,i,ii,iii);
                disp('****************************************')
                disp('ACCmean =')
                disp(ACCmean(i,ii,iii))
                disp('NMImean =')
                disp(NMImean(i,ii,iii))
                disp('Cohmean =')
                disp(Cohmean(i,ii,iii))
                disp('ARImean =')
                disp(ARImean(i,ii,iii))
            end
        end
    end
    
    ACCfinalmean=max(ACCmean(:));
    NMIfinalmean=max(NMImean(:));
    Cohfinalmean=max(Cohmean(:));
    ARIfinalmean=max(ARImean(:));

    finalResults = [ACCfinalmean;NMIfinalmean;Cohfinalmean;ARIfinalmean];
    % xlswrite('myResults.xlsx',finalResults,'ONMF-H','D18:D21')
    SS2 =['save Finalresults_' num2str(NT) '_Topic   finalResults;'];
    eval(SS2);
    fprintf(1,'For #Tpoics = %d , The final results are: \n',NT)
    disp('finalACC =')
    disp(ACCfinalmean)
    disp('finalNMI =')
    disp(NMIfinalmean)
    disp('finalCoh =')
    disp(Cohfinalmean)
    disp('finalARI =')
    disp(ARIfinalmean)
    disp('****************************************')

end

