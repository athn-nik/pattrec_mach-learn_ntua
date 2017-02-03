clear all;
close all;
load('finalAct.mat');
load('finalVal.mat');
load('MIRFeatures.mat');

%% Step 10

% Ignore samples with value = 3
%keepSamples = intersect(find(finalActivation ~= 3),find(finalValence ~= 3));
Act_kept = find(finalAct ~= 3);
Val_kept = find(finalVal ~= 3);
Act = finalAct(Act_kept);
Val = finalVal(Val_kept);
samplesremained_act = size(Act,2);
samplesremained_val = size(Val,2);
kept_values=[samplesremained_act samplesremained_val];

%plot the values kept of each category

x=1:2;
bar(x,kept_values,.2)
Labels = {'Activation', 'Valence'};
set(gca, 'XTick', 1:2, 'XTickLabel', Labels);
for i=1:2
    text(x(i),kept_values(i),num2str(kept_values(i),'%0.0f'),'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end

% Fix the arrays to treshold for high low activation and positive-negative
% valence
% High Activation
Act(Act < 3) = 1.0; 
% Low Activation
Act(Act > 3) = -1.0;
% Negative Valence
Val(Val < 3) = -1.0;
 % Positive Valence
Val(Val > 3) = 1.0;        

%% Step 11


% Preprocessing MIR Features
 
%keep only the elements we want
MIRFeatures = cell2mat(MIRFeatures);
MIRFeatures_act = MIRFeatures(Act_kept,:); 
MIRFeatures_val = MIRFeatures(Val_kept,:); 

for times = 1:3
        % Split to random Training Set 80% and Validation Set 20%
        randmix_act = randperm(samplesremained_act);
        randmix_val = randperm(samplesremained_val);
        MIRFeatures_act = MIRFeatures_act(randmix_act,:);
        MIRFeatures_val = MIRFeatures_val(randmix_val,:);
        Act = Act(randmix_act);
        Val = Val(randmix_val);

        %mix elements and fix the final class
        Act_step6 = [MIRFeatures_act(:,1:10) Act'];
        Val_step6 = [MIRFeatures_val(:,1:10) Val'];
        
        Act_step7 = [MIRFeatures_act(:,11:end) Act'];
        Val_step7 = [MIRFeatures_val(:,11:end) Val'];

        %% Step 14 - PCA (Inserted in Step 11 as it has to do with preprocessing of data)
%          change the p to vary the number of components
%          p = 60;
%          [~,MIRFeatures_act,~] = pca(MIRFeatures_act, 'NumComponents',p);
%          [~,MIRFeatures_val,~] = pca(MIRFeatures_val, 'NumComponents',p);

        Act_all = [MIRFeatures_act Act'];
        Val_all = [MIRFeatures_val Val'];
        %calculate the size of validation set 
        valset_a = floor(samplesremained_act/5);
        valset_v = floor(samplesremained_val/5);

        %partition each category of experimentazation in 4/5train and 1/5
        %test
        
        valset_a_step6 = Act_step6(1:valset_a,:);
        trset_a_step6 = Act_step6(valset_a+1:end,:);
        valset_v_step6 = Val_step6(1:valset_v,:);
        trset_v_step6 = Val_step6(valset_v+1:end,:);

        valset_a_step7 = Act_step7(1:valset_a,:);
        trset_a_step7 = Act_step7(valset_a+1:end,:);
        valset_v_step7 = Val_step7(1:valset_v,:);
        trset_v_step7 = Val_step7(valset_v+1:end,:);

        valset_a_all = Act_all(1:valset_a,:);
        trset_a_all = Act_all(valset_a+1:end,:);
        valset_v_all = Val_all(1:valset_v,:);
        trset_v_all = Val_all(valset_v+1:end,:);
% final column is used to know the size of validation set for passing it as
% a parameter to metrics for correct accuracy and f1 results
        finalcolumn1 = size(valset_a_step6,2);
        finalcolumn2 = size(valset_a_step7,2);
        finalcolumn3 = size(valset_a_all,2);

        %% Step 12 - NNR
        for neighbors = 1:2:7
            act_nnr_step6 = nnr_k(trset_a_step6,valset_a_step6,neighbors);
            [acc_act_nnr_s6(times,neighbors),prec_act_nnr_s6(times,neighbors),rec_act_nnr_s6(times,neighbors),f1_act_nnr_s6(times,neighbors)] = ...
                metrics(valset_a_step6(:,finalcolumn1),act_nnr_step6);
            act_nnr_step7 = nnr_k(trset_a_step7,valset_a_step7,neighbors);
            [acc_act_nnr_s7(times,neighbors),prec_act_nnr_s7(times,neighbors),rec_act_nnr_s7(times,neighbors),f1_act_nnr_s7(times,neighbors)] = ...
                metrics(valset_a_step7(:,finalcolumn2),act_nnr_step7);
            act_nnr_all = nnr_k(trset_a_all,valset_a_all,neighbors);
            [acc_act_nnr_all(times,neighbors),prec_act_nnr_all(times,neighbors),rec_act_nnr_all(times,neighbors),f1_act_nnr_all(times,neighbors)] = ...
                metrics(valset_a_all(:,finalcolumn3),act_nnr_all);

            
            val_nnr_step6 = nnr_k(trset_v_step6,valset_v_step6,neighbors);
            [acc_val_nnr_s6(times,neighbors),prec_val_nnr_s6(times,neighbors),rec_val_nnr_s6(times,neighbors),f1_val_nnr_s6(times,neighbors)] = ...
                metrics(valset_v_step6(:,finalcolumn1),val_nnr_step6);
            val_nnr_step7 = nnr_k(trset_v_step7,valset_v_step7,neighbors);
            [acc_val_nnr_s7(times,neighbors),prec_val_nnr_s7(times,neighbors),rec_val_nnr_s7(times,neighbors),f1_val_nnr_s7(times,neighbors)] = ...
                metrics(valset_v_step7(:,finalcolumn2),val_nnr_step7);
            val_nnr_all = nnr_k(trset_v_all,valset_v_all,neighbors);
            [acc_val_nnr_all(times,neighbors),prec_val_nnr_all(times,neighbors),rec_val_nnr_all(times,neighbors),f1_val_nnr_all(times,neighbors)] = ...
                metrics(valset_v_all(:,finalcolumn3),val_nnr_all);
        end
        %% Step 13 -  Bayes
        act_bayes_s6 = bayes(trset_a_step6,valset_a_step6);
         [acc_act_bayes_s6(times),prec_act_bayes_s6(times),rec_act_bayes_s6(times),f1_act_bayes_s6(times)] = ...
                metrics(valset_a_step6(:,finalcolumn1),act_bayes_s6);
        act_bayes_s7 = bayes(trset_a_step7,valset_a_step7);
         [acc_act_bayes_s7(times),prec_act_bayes_s7(times),rec_act_bayes_s7(times),f1_act_bayes_s7(times)] = ...
                metrics(valset_a_step7(:,finalcolumn2),act_bayes_s7);
        act_bayes_all = bayes(trset_a_all,valset_a_all);
         [acc_act_bayes_all(times),prec_act_bayes_all(times),rec_act_bayes_all(times),f1_act_bayes_all(times)] = ...
                metrics(valset_a_all(:,finalcolumn3),act_bayes_all);

            
        val_bayes_s6 = bayes(trset_v_step6,valset_v_step6);
         [acc_val_bayes_s6,prec_val_bayes_s6,rec_val_bayes_s6,f1_val_bayes_s6] = ...
                metrics(valset_v_step6(:,finalcolumn1),val_bayes_s6);
        val_bayes_s7 = bayes(trset_v_step7,valset_v_step7);
         [acc_val_bayes_s7,prec_val_bayes_s7,rec_val_bayes_s7,f1_val_bayes_s7] = ...
                metrics(valset_v_step7(:,finalcolumn2),val_bayes_s7);
        val_bayes_all = bayes(trset_v_all,valset_v_all);
         [acc_val_bayes_all,prec_val_bayes_all,rec_val_bayes_all,f1_val_bayes_all] = ...
                metrics(valset_v_all(:,finalcolumn3),val_bayes_all);

end

%mean for activation for all 3 set of features and classifiers
mean_accuracy_act_nnr_s6 = mean(acc_act_nnr_s6);
mean_precision_act_nnr_s6 =mean(prec_act_nnr_s6);
mean_recall_act_nnr_s6 = mean(rec_act_nnr_s6);
mean_f1_act_nnr_s6 = mean(f1_act_nnr_s6);
mean_accuracy_act_bayes_s6 = mean(acc_act_bayes_s6);
mean_precision_act_bayes_s6 =mean(prec_act_bayes_s6);
mean_recall_act_bayes_s6 = mean(rec_act_bayes_s6);
mean_f1_act_bayes_s6 = mean(f1_act_bayes_s6);


mean_accuracy_act_nnr_s7 = mean(acc_act_nnr_s7);
mean_precision_act_nnr_s7 =mean(prec_act_nnr_s7);
mean_recall_act_nnr_s7 = mean(rec_act_nnr_s7);
mean_f1_act_nnr_s7 = mean(f1_act_nnr_s7);
mean_accuracy_act_bayes_s7 = mean(acc_act_bayes_s7);
mean_precision_act_bayes_s7 =mean(prec_act_bayes_s7);
mean_recall_act_bayes_s7 = mean(rec_act_bayes_s7);
mean_f1_act_bayes_s7 = mean(f1_act_bayes_s7);


mean_accuracy_act_nnr_all = mean(acc_act_nnr_all);
mean_precision_act_nnr_all =mean(prec_act_nnr_all);
mean_recall_act_nnr_all = mean(rec_act_nnr_all);
mean_f1_act_nnr_all = mean(f1_act_nnr_all);
mean_accuracy_act_bayes_all = mean(acc_act_bayes_all);
mean_precision_act_bayes_all =mean(prec_act_bayes_all);
mean_recall_act_bayes_all = mean(rec_act_bayes_all);
mean_f1_act_bayes_all = mean(f1_act_bayes_all);

%mean for valence for all 3 sets of features for all classifiers

mean_accuracy_val_nnr_s6 = mean(acc_val_nnr_s6);
mean_precision_val_nnr_s6 =mean(prec_val_nnr_s6);
mean_recall_val_nnr_s6 = mean(rec_val_nnr_s6);
mean_f1_val_nnr_s6 = mean(f1_val_nnr_s6);
mean_accuracy_val_bayes_s6 = mean(acc_val_bayes_s6);
mean_precision_val_bayes_s6 =mean(prec_val_bayes_s6);
mean_recall_val_bayes_s6 = mean(rec_val_bayes_s6);
mean_f1_val_bayes_s6 = mean(f1_val_bayes_s6);


mean_accuracy_val_nnr_s7 = mean(acc_val_nnr_s7);
mean_precision_val_nnr_s7 =mean(prec_val_nnr_s7);
mean_recall_val_nnr_s7 = mean(rec_val_nnr_s7);
mean_f1_val_nnr_s7 = mean(f1_val_nnr_s7);
mean_accuracy_val_bayes_s7 = mean(acc_val_bayes_s7);
mean_precision_val_bayes_s7 =mean(prec_val_bayes_s7);
mean_recall_val_bayes_s7 = mean(rec_val_bayes_s7);
mean_f1_val_bayes_s7 = mean(f1_val_bayes_s7);


mean_accuracy_val_nnr_all = mean(acc_val_nnr_all);
mean_precision_val_nnr_all =mean(prec_val_nnr_all);
mean_recall_val_nnr_all = mean(rec_val_nnr_all);
mean_f1_val_nnr_all = mean(f1_val_nnr_all);
mean_accuracy_val_bayes_all = mean(acc_val_bayes_all);
mean_precision_val_bayes_all =mean(prec_val_bayes_all);
mean_recall_val_bayes_all = mean(rec_val_bayes_all);
mean_f1_val_bayes_all = mean(f1_val_bayes_all);


%% Step 15 

%you don't have to run this that's why it is commented
%the arff files is in the zip file 


%{

javaaddpath('C:\program files\Weka-3-6\weka.jar')
addpath('./matlab2weka');

%until last-1 element fix the array of string as the function matlab2weka
%asks for final column is string of class after that we create and save
%arff file 
d = {};
for i = 1:size(Act_step6,2)-1
    d{i} = ['feature',num2str(i)];
end
d{size(Act_step6,2)} = 'Activation';
WekaInstance = matlab2weka('ActivationWekas6',d,Act_step6);
saveARFF('act_6.arff',WekaInstance);

d = {};
for i = 1:size(Act_step7,2)-1
    d{i} = ['feature',num2str(i)];
end
d{size(Act_step7,2)} = 'Activation';
WekaInstance = matlab2weka('Activation_Weka_s7',d,Act_step7);
saveARFF('act_7.arff',WekaInstance);

d = {};
for i = 1:size(Act_all,2)-1
    d{i} = ['feature',num2str(i)];
end
d{size(Act_all,2)} = 'Activation';
WekaInstance = matlab2weka('Activation_Weka_all',d,Act_all);
saveARFF('act_all.arff',WekaInstance);

d = {};
for i = 1:size(Val_step6,2)-1
    d{i} = ['feature',num2str(i)];
end
d{size(Val_step6,2)} = 'Valence';
WekaInstance = matlab2weka('Valence_Weka_s6',d,Val_step6);
saveARFF('val_6.arff',WekaInstance);

d = {};
for i = 1:size(Val_step7,2)-1
    d{i} = ['feature',num2str(i)];
end
d{size(Val_step7,2)} = 'Valence';
WekaInstance = matlab2weka('Valence_Weka_s7',d,Val_step7);
saveARFF('val_7.arff',WekaInstance);

%until the last-1 element(because last is class)
d = {};
for i = 1:size(Val_all,2)-1
    d{i} = ['feature',num2str(i)];
end
%last-1 is class
d{size(Val_all,2)} = 'Valence';
WekaInstance = matlab2weka('Valence_Weka_all',d,Val_all);
saveARFF('val_all.arff',WekaInstance);

%}
