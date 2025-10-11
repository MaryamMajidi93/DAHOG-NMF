# Compute Adjusted Rand Index (ARI) measure

function [ARI] = ARIMea(labels1, labels2)

try
    conf = confusionmat(labels1, labels2);  % Use built-in function to generate confusion matrix
catch
    conf = myConfusionMat(labels1, labels2);  % If built-in function is not available, use the self-defined function
end

% Compute necessary quantities for ARI calculation
sum_C = sum(conf(:));
sum_C2 = sum_C * (sum_C - 1);
sum_rows = sum(conf, 2);
sum_rows2 = sum(sum_rows .* (sum_rows - 1));
sum_cols = sum(conf, 1);
sum_cols2 = sum(sum_cols .* (sum_cols - 1));
sum_Cij2 = sum(sum(conf .* (conf - 1)));

% Compute ARI
ARI = 2 * (sum_Cij2 - sum_rows2 * sum_cols2 / sum_C2) / ...
    ((sum_rows2 + sum_cols2) - 2 * sum_rows2 * sum_cols2 / sum_C2);

if ARI < 0
    ARI = 0;
end

end


function C = myConfusionMat(g1, g2)
% This function generates a confusion matrix if the built-in function confusionmat is not available.
% It takes two input vectors, g1 and g2, which represent two different label assignments.

groups = unique([g1;g2]);  % Get the unique groups in g1 and g2

C = zeros(length(groups));  % Initialize the confusion matrix with zeros

% Calculate the confusion matrix
for i = 1:length(groups)
    for j = 1:length(groups)
        C(i,j) = sum(g1 == groups(i) & g2 == groups(j));  % Count the number of samples that belong to group i in g1 and group j in g2
    end
end

end


