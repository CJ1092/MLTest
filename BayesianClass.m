%% Bayesian Classification with and without Naive assumptions

clear all;close all;

%% Load the data

load cbt1data;
train_data = [healthy; diseased]; %Combining the two training dataset.
s1 = size(healthy,1); s2 = size(diseased,1);
%we use 1 and 2 as class identification for easier references
t2 = ones(s1,1)+1; %class 1 for healthy(later becomes 2).
t1 = zeros(s2,1)+1; %class 0 for diseased(later becomes 1).

%% Using the Naive assumption

for c = 1:length(cl)
pos = find(target==cl(c));
% Find the means and variances
class_mean(c,:) = mean(train_data(pos,:)); % class-wise & attribute-wise mean of training data
class_var(c,:) = var(train_data(pos,:),1); % class-wise & attribute-wise variance of training data
end

%% Compute the predictive probabilities (with Naive assumption)

probab_train = [];
for c = 1:length(cl)
sigmac1 = diag(class_var(c,:));

diff_train = [train_data(:,1)-class_mean(c,1) train_data(:,2)-class_mean(c,2)];
const_train = 1/sqrt((2*pi)^size(train_data,2) * det(sigmac1));
%Gaussian class conditional probability for training data with Naive
probab_train(:,c) = const_train*exp(-0.5*diag(diff_train*inv(sigmac1)*diff_train'));
ML_est_with = probab_train;
MAP_est_h_with = probab_train*P_Heal;
MAP_est_d_with = probab_train*P_Dis;

end

MAP_est_with = MAP_est_h_with + MAP_est_d_with;
% proper probability estimates
probab_train_orig = probab_train./repmat(sum(probab_train,2),[1,2]);
MAP_est_with_orig = MAP_est_with./repmat(sum(MAP_est_with,2),[1,2]);
%% class label predictions from probabilities (with Naive assumption)

[~,p_train_with] = max(probab_train_orig,[],2); % assign labels as per highest probability
compare_train_with=[target p_train_with target-p_train_with]; % for comparison
error_train_with=sum(target~=p_train_with); % error - # of mis-classifications

[~,pmap_train_with] = max(MAP_est_with_orig,[],2); % assign labels as per highest probability
comparemap_train_with=[target pmap_train_with target-pmap_train_with]; % for comparison
errormap_train_with=sum(target~=pmap_train_with); % error - # of mis-classifications


%% without using Naive assumption
class_mean_wo = [];
class_var_wo = [];
for c = 1:length(cl)
pos = find(target==cl(c));
% Find the means and covariances
class_mean_wo(c,:) = mean(train_data(pos,:)); % class-wise & attribute-wise mean of training data
class_var_wo(:,:,c) = cov(train_data(pos,:),1); % class-wise & attribute-wise co-variance of training data
end

%% Compute the predictive probabilities (without Naive assumption)

probab_train_wo = [];

for c = 1:length(cl)
sigmac2 = class_var_wo(:,:,c);

diff_train_wo = [train_data(:,1)-class_mean(c,1) train_data(:,2)-class_mean(c,2)];
const_train_wo = 1/sqrt((2*pi)^size(train_data,2) * det(sigmac2));
%Gaussian class conditional probability for training data without Naive
probab_train_wo(:,c) = const_train_wo*exp(-0.5*diag(diff_train_wo*inv(sigmac2)*diff_train_wo'));
ML_est_wo = probab_train_wo;
MAP_est_h_wo = probab_train_wo*P_Heal;
MAP_est_d_wo = probab_train_wo*P_Dis;

end

MAP_est_wo = MAP_est_h_wo + MAP_est_d_wo;
% proper probability estimates
probab_train_wo_orig = probab_train_wo./repmat(sum(probab_train_wo,2),[1,2]);
MAP_est_wo_orig = MAP_est_wo./repmat(sum(MAP_est_wo,2),[1,2]);
%% class label predictions from probabilities (without Naive assumption)
[~,p_train_without] = max(probab_train_wo_orig,[],2); % assign labels as per highest probability
compare_train_without=[target p_train_without target-p_train_without]; % for comparison
error_train_without=sum(target~=p_train_without); % error - # of mis-classifications

[~,pmap_train_wo] = max(MAP_est_wo_orig,[],2); % assign labels as per highest probability
comparemap_train_wo=[target pmap_train_wo target-pmap_train_wo]; % for comparison
errormap_train_wo=sum(target~=pmap_train_wo); % error - # of mis-classifications
