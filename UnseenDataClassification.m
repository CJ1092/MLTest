%% Unseen Data class Prediction
%% With Naive Assumptions
probab_train_unseen = [];
for c = 1:length(cl)
    sigmac1 = diag(class_var(c,:));
    
    diff_train_u = [unseen(:,1)-class_mean(c,1) unseen(:,2)-class_mean(c,2)];
    const_train_u = 1/sqrt((2*pi)^size(unseen,2) * det(sigmac1));
    %Gaussian class conditional probability for training data with Naive
    probab_train_unseen(:,c) = const_train_u*exp(-0.5*diag(diff_train_u*inv(sigmac1)*diff_train_u'));
    ML_est_with_u = probab_train_unseen;
    MAP_est_h_with_u = probab_train_unseen*P_Heal;
    MAP_est_d_with_u = probab_train_unseen*P_Dis;
   
end
MAP_est_u = MAP_est_h_with_u + MAP_est_d_with_u;
 
% probabilities
ML_est_with_u_orig = ML_est_with_u./repmat(sum(ML_est_with_u,2),[1,2]);
MAP_est_u_orig = MAP_est_u./repmat(sum(MAP_est_u,2),[1,2]);
 
%% class label predictions from probabilities of unseen data (with Naive assumption)
 
[~,p_train_with_u] = max(ML_est_with_u_orig,[],2); % assign labels as per highest probability
 
[~,pmap_train_with_u] = max(MAP_est_u_orig,[],2); % assign labels as per highest probability
 
%% Without Naive Assumptions
 
probab_train_wo_unseen = [];
 
for c = 1:length(cl)
    sigmac2 = class_var_wo(:,:,c); 
    
    diff_train_wo_u = [unseen(:,1)-class_mean(c,1) unseen(:,2)-class_mean(c,2)];
    const_train_wo_u = 1/sqrt((2*pi)^size(unseen,2) * det(sigmac2));
    %Gaussian class conditional probability for training data without Naive
    probab_train_wo_unseen(:,c) = const_train_wo_u*exp(-0.5*diag(diff_train_wo_u*inv(sigmac2)*diff_train_wo_u'));
    ML_est_wo_u = probab_train_wo_unseen;
    MAP_est_h_wo_u = probab_train_wo_unseen*P_Heal;
    MAP_est_d_wo_u = probab_train_wo_unseen*P_Dis;
    
end
 
MAP_est_wo_u = MAP_est_h_wo_u + MAP_est_d_wo_u;
% proper probability estimates
ML_est_wo_u_orig = ML_est_wo_u./repmat(sum(ML_est_wo_u,2),[1,2]);
MAP_est_wo_u_orig = MAP_est_wo_u./repmat(sum(MAP_est_wo_u,2),[1,2]);
 
%% class label predictions from probabilities of unseen data (without Naive assumption)
 
[~,p_train_wo_u] = max(ML_est_wo_u_orig,[],2); % assign labels as per highest probability
 
[~,pmap_train_wo_u] = max(MAP_est_wo_u_orig,[],2); % assign labels as per highest probability
