clear all;
clc;

% TODO:
% - implement CHL
% - modularize code
% - play around with activation noise

% experiment parameters

% number of pairs to be memorized
Npairs=10;
decayIterations_postStudy = 100;
decayIterations_postReStudy = 0; 
decayIterations_postTest = decayIterations_postReStudy;
decayIterations_forgettingCurve = 100; 

% try different tresholds / try different number of max. iterations foe study vs. group manipulations

% network configuration

% network parameters
Nunits = 2*Npairs;      % number of units
N_threshold = 2;         % # of units required to reach threshold
w_init_scale = 0.01;    % scale of initial weights
gain = 1.5;                     % gain of activation function
tau = 0.05;                  % time integration constant
eta = 1;                        % BCM learning rate
decayRate = 0.001;        % weight decay rate
decayNoise = 0.00;      % weight decay noise
threshold_Study = 0.20;                  % integration threshold (between 0 and 1)
threshold_postStudy = threshold_Study;          % integration threshold (between 0 and 1)
inputStrength = 1;      % strength of input
maxTimeSteps_Study = 500; % maximum number of time steps
maxTimeSteps_postStudy = maxTimeSteps_Study; % maximum number of time steps
batch_learning = 1;     % set to 1 if weights should be adjustment in batch mode (in that case, the order of stimuli in retrieval phases does not matter)

% initial activation
Act_init = zeros(Nunits, 1);

% generate weight matrix
W_init = rand(Nunits) * w_init_scale - w_init_scale/2;

% create memory network
memoryNet_study = simpleMemoryNet(W_init, Act_init, threshold_Study, gain, tau, 'BCM');
memoryNet_study.maxTimeSteps = maxTimeSteps_Study;
memoryNet_study.eta = eta;

% initial study

% generate input for study phase
studyInput=zeros(Npairs,Nunits);
for pattern=1:Npairs
    studyInput(pattern,pattern)=inputStrength;
    studyInput(pattern, pattern+Npairs)=inputStrength;
end


% for each pattern
for pattern = 1:Npairs
    
    if(batch_learning == 1)
        resetActivationLog = 0;
    else
        resetActivationLog = 1;
    end
    
    % determine current input
    input = studyInput(pattern, :);
    
    % let network settle until threshold
    study_activationLog{pattern} = memoryNet_study.runTrialUntilThreshold(input, N_threshold, resetActivationLog);
    
    if(batch_learning == 0)
        % adjust weights
        [W, fract] = memoryNet_study.adjustWeights();
    end
    
end

if(batch_learning == 1)
        % adjust weights
        [W, fract] = memoryNet_study.adjustWeights();
end

% log weights
W_afterStudy = memoryNet_study.W;
% store study net to save deep copy
save('studyNet_tmp.mat', 'memoryNet_study');

% weight decay after study phase
memoryNet_study.decayWeights(decayRate, decayIterations_postStudy, decayNoise);
W_afterStudyDecay = memoryNet_study.W;

% change response threshold and number of time steps for post-study phases
memoryNet_study.threshold = threshold_postStudy;
memoryNet_study.maxTimeSteps = maxTimeSteps_postStudy;

% restudy group

% generate net for restudy condition
memoryNet_restudyGroup = memoryNet_study;

% for each pattern
for pattern = 1:Npairs
    
    if(batch_learning == 1 && pattern ~= 1)
        resetActivationLog = 0;
    else
        resetActivationLog = 1;
    end
    
    % determine current input
    input = studyInput(pattern, :);
    
    % let network settle until threshold
    restudy_activationLog{pattern} = memoryNet_restudyGroup.runTrialUntilThreshold(input, N_threshold, resetActivationLog);
    
    if(~batch_learning)
        % adjust weights
        [W,fract] = memoryNet_restudyGroup.adjustWeights();
    end
    
end

if(batch_learning)
    % adjust weights
        [W,fract] = memoryNet_restudyGroup.adjustWeights();
end

% log weights
W_afterReStudy = memoryNet_restudyGroup.W;

% test goup

% generate net for restudy condition
load('studyNet_tmp.mat');
memoryNet_testGroup = memoryNet_study;
accuracy_test = nan(1, Npairs);
RT_test = nan(1, Npairs);

input_log = [];

% for each pattern
for pattern = 1:Npairs
    
    if(batch_learning == 1  && pattern ~= 1)
        resetActivationLog = 0;
    else
        resetActivationLog = 1;
    end
    
    % determine current input
    input = studyInput(pattern, :);
    input((Npairs+1):end) = 0;
    correct = studyInput(pattern, :);
    input_log = [input_log; input];
    
    % let network settle until threshold
    test_activationLog{pattern} = memoryNet_testGroup.runTrialUntilThreshold(input, N_threshold, resetActivationLog);
    
    % compute accuracy & RT
    accuracy_test(pattern) = memoryNet_testGroup.computeAccuracy(correct);
    RT_test(pattern) = length(test_activationLog{pattern});
    
    if(~batch_learning)
        % adjust weights
        [W,fract] = memoryNet_testGroup.adjustWeights();
    end
    
end

if(batch_learning)
    % log weights
    W_afterTest = memoryNet_testGroup.W;
end


% final test

% weight decay after restudy/test phase
memoryNet_restudyGroup.decayWeights(decayRate, decayIterations_postReStudy, decayNoise);
W_afterReStudyDecay = memoryNet_restudyGroup.W;

memoryNet_testGroup.decayWeights(decayRate, decayIterations_postTest, decayNoise);
W_afterTestDecay = memoryNet_testGroup.W;

% run final test (without learning)

accuracy_finalTest_restudyGroup = nan(1, Npairs);
RT_finalTest_restudyGroup = nan(1, Npairs);
accuracy_finalTest_testGroup = nan(1, Npairs);
RT_finalTest_testGroup = nan(1, Npairs);

% for each pattern
for pattern = 1:Npairs
    
    % determine current input
    input = studyInput(pattern, :);
    input((Npairs+1):end) = 0;
    correct = studyInput(pattern, :);
    
    
    % study network
    
    % let network settle until threshold
    finalTest_restudyGroup_activationLog{pattern} = memoryNet_restudyGroup.runTrialUntilThreshold(input, N_threshold);
    
    % compute accuracy & RT
   accuracy_finalTest_restudyGroup(pattern) = memoryNet_restudyGroup.computeAccuracy(correct);
   RT_finalTest_restudyGroup(pattern) = length(finalTest_restudyGroup_activationLog{pattern});
    
   % test network
   
    % let network settle until threshold
    finalTest_testGroup_activationLog{pattern} = memoryNet_testGroup.runTrialUntilThreshold(input, N_threshold);
    
    % compute accuracy & RT
    accuracy_finalTest_testGroup(pattern) = memoryNet_testGroup.computeAccuracy(correct);
    RT_finalTest_testGroup(pattern) = length(finalTest_testGroup_activationLog{pattern});
    
end



% plot

plotPattern = 1;

allWeights = [W_afterStudy(:); W_init(:); W_afterStudyDecay(:); W_afterReStudyDecay(:); W_afterTestDecay(:); W_afterReStudy(:); W_afterTest(:)];
w_limit = [min(allWeights) max(allWeights)];
allActs = [study_activationLog{plotPattern}(:); ...
                restudy_activationLog{plotPattern}(:); ...
                test_activationLog{plotPattern}(:); ...
                finalTest_restudyGroup_activationLog{plotPattern}(:); ...
                finalTest_testGroup_activationLog{plotPattern}(:)];
act_limit = [min(allActs) max(allActs)];            

fig = figure(1);
set(fig, 'Position', [100 100 1300 600]);

% initialization
subplot(2,7,8);
imagesc(W_init); colorbar;
caxis([w_limit]);
title({'initial',  'weights'});

% study
subplot(2,7,2);
plot(study_activationLog{plotPattern}); 
ylim(act_limit);
xlabel('time');
ylabel('activation')
title('study');

subplot(2,7,9);
imagesc(W_afterStudy);  
caxis([w_limit]);
title({'weights after',  'study'});

% decay after study
subplot(2,7,10);
imagesc(W_afterStudyDecay);  
caxis([w_limit]);
title({'weights after',  'decay'});

% restudy
subplot(2,7,4);
plot(restudy_activationLog{plotPattern}); 
ylim(act_limit);
xlabel('time');
ylabel('activation')
title('re-study phase');

subplot(2,7,11);
imagesc(W_afterReStudy);  
caxis([w_limit]);
title({'weights after', 'restudy phase'});

% test
subplot(2,7,5);
plot(test_activationLog{plotPattern}); 
ylim(act_limit);
xlabel('time');
ylabel('activation')
title('test phase');

subplot(2,7,12);
imagesc(W_afterTest);  
caxis([w_limit]);
title({'weights after', 'test phase'});

% test rerestudy group
subplot(2,7,6);
plot(finalTest_restudyGroup_activationLog{plotPattern});
ylim(act_limit);
xlabel('time');
ylabel('activation')
accuracy = mean(accuracy_finalTest_restudyGroup) * 100;
RT = mean(RT_finalTest_restudyGroup) * 100;
title({'final test (restudy group)' ...
        ['Acc: ' num2str(accuracy) '%, RT = ' ...
         num2str(RT)]});

subplot(2,7,13);
imagesc(W_afterReStudyDecay);  
caxis([w_limit]);
title({'weights at final test', '(restudy group)'});

% test test group
subplot(2,7,7);
plot(finalTest_testGroup_activationLog{plotPattern}); 
ylim(act_limit);
xlabel('time');
ylabel('activation')
accuracy = mean(accuracy_finalTest_testGroup) * 100;
RT = mean(RT_finalTest_testGroup) * 100;
title({'final test (test group)' ...
        ['Acc: ' num2str(accuracy) '%, RT = ' ...
         num2str(RT)]});

subplot(2,7,14);
imagesc(W_afterTestDecay);  
caxis([w_limit]);
title({'weights at final test', '(test group)'});



%% FORGETTING PHASE
accuracy_log_restudyGroup = nan(1, decayIterations_forgettingCurve);
accuracy_log_testGroup = nan(1, decayIterations_forgettingCurve);
RT_log_restudyGroup = nan(1, decayIterations_forgettingCurve);
RT_log_testGroup = nan(1, decayIterations_forgettingCurve);

for iter = 1:decayIterations_forgettingCurve
    
    disp(iter);
    
    % weight decay
    memoryNet_restudyGroup.decayWeights(decayRate, 1, decayNoise);
    memoryNet_testGroup.decayWeights(decayRate, 1, decayNoise);
    
    % for each pattern
    for pattern = 1:Npairs

        % determine current input
        input = studyInput(pattern, :);
        input((Npairs+1):end) = 0;
        correct = studyInput(pattern, :);


        % study network

        % let network settle until threshold
        finalTest_restudyGroup_activationLog{pattern} = memoryNet_restudyGroup.runTrialUntilThreshold(input, N_threshold);

        % compute accuracy & RT
       accuracy_finalTest_restudyGroup(pattern) = memoryNet_restudyGroup.computeAccuracy(correct);
       RT_finalTest_restudyGroup(pattern) = length(finalTest_restudyGroup_activationLog{pattern});

       % test network

        % let network settle until threshold
        finalTest_testGroup_activationLog{pattern} = memoryNet_testGroup.runTrialUntilThreshold(input, N_threshold);

        % compute accuracy & RT
        accuracy_finalTest_testGroup(pattern) = memoryNet_testGroup.computeAccuracy(correct);
        RT_finalTest_testGroup(pattern) = length(finalTest_testGroup_activationLog{pattern});

    end
    
    accuracy_log_restudyGroup(iter) = mean(accuracy_finalTest_restudyGroup);
    accuracy_log_testGroup(iter) = mean(accuracy_finalTest_testGroup);
    RT_log_restudyGroup(iter) = mean(RT_finalTest_restudyGroup);
    RT_log_testGroup(iter) = mean(RT_finalTest_testGroup);
    
end

% PLOTS 

figure(2);
subplot(1,2,1);
plot(accuracy_log_restudyGroup*100, '-b', 'LineWidth', 3); hold on;
plot(accuracy_log_testGroup*100, '-r', 'LineWidth', 3); hold off;
legend('restudy group', 'test group');
xlabel('time');
ylabel('accuracy (%)');

subplot(1,2,2);
plot(RT_log_restudyGroup, '-b', 'LineWidth', 3); hold on;
plot(RT_log_testGroup, '-r', 'LineWidth', 3); hold off;
legend('restudy group', 'test group');
xlabel('time');
ylabel('RT (s)');
