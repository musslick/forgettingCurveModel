clear all;
clc;

% experiment parameters

% number of pairs to be memorized
Npairs=10;
decayIterations_postStudy = 1000;
decayIterations_postReStudy = 1000;
decayIterations_postTest = decayIterations_postReStudy;

% network configuration

% network parameters
Nunits = 2*Npairs;      % number of units
N_threshold = 2;         % # of units required to reach threshold
w_init_scale = 0.01;    % scale of initial weights
gain = 1;                     % gain of activation function
tau = 0.1;                  % time integration constant
eta = 1;                  % BCM learning rate
decayRate = 0.01;   % weight decay rate
decayNoise = 0.00;  % weight decay noise
threshold = 0.2;          % integration threshold (between 0 and 1)
inputStrength = 1;      % strength of input
maxTimeSteps = 50; % maximum number of time steps

% initial activation
Act_init = zeros(Nunits, 1);

% generate weight matrix
W_init = rand(Nunits) * w_init_scale - w_init_scale/2;

% create memory network
memoryNet_study = simpleMemoryNet(W_init, Act_init, threshold, gain, tau, 'BCM');
memoryNet_study.maxTimeSteps = maxTimeSteps;
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
    
    % determine current input
    input = studyInput(pattern, :);
    
    % let network settle until threshold
    study_activationLog{pattern} = memoryNet_study.runTrialUntilThreshold(input, N_threshold);
    
    % adjust weights
    [W,fract] = memoryNet_study.adjustWeights();
    
end

% log weights
W_afterStudy = memoryNet_study.W;
% store study net to save deep copy
save('studyNet_tmp.mat', 'memoryNet_study');

% weight decay after study phase
memoryNet_study.decayWeights(decayRate, decayIterations_postStudy, decayNoise);
W_afterStudyDecay = memoryNet_study.W;

% restudy phase

% generate net for restudy condition
memoryNet_restudyGroup = memoryNet_study;

% for each pattern
for pattern = 1:Npairs
    
    % determine current input
    input = studyInput(pattern, :);
    
    % let network settle until threshold
    restudy_activationLog{pattern} = memoryNet_restudyGroup.runTrialUntilThreshold(input, N_threshold);
    
    % adjust weights
    [W,fract] = memoryNet_restudyGroup.adjustWeights();
    
end

% log weights
W_afterReStudy = memoryNet_restudyGroup.W;

% test phase

% generate net for restudy condition
load('studyNet_tmp.mat');
memoryNet_testGroup = memoryNet_study;
accuracy_test = nan(1, Npairs);
RT_test = nan(1, Npairs);

% for each pattern
for pattern = 1:Npairs
    
    % determine current input
    input = studyInput(pattern, :);
    input(Npairs+1) = 0;
    correct = [pattern pattern+Npairs];
    
    % let network settle until threshold
    test_activationLog{pattern} = memoryNet_testGroup.runTrialUntilThreshold(input, N_threshold);
    
    % compute accuracy & RT
    accuracy_test(pattern) = memoryNet_testGroup.computeAccuracy(input);
    RT_test = length(test_activationLog{pattern});
    
    % adjust weights
    [W,fract] = memoryNet_testGroup.adjustWeights();

end

% log weights
W_afterTest = memoryNet_testGroup.W;

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
    input(Npairs+1) = 0;
    correct = [pattern pattern+Npairs];
    
    % study network
    
    % let network settle until threshold
    finalTest_restudyGroup_activationLog{pattern} = memoryNet_restudyGroup.runTrialUntilThreshold(input, N_threshold);
    
    % compute accuracy & RT
   accuracy_finalTest_restudyGroup(pattern) = memoryNet_restudyGroup.computeAccuracy(input);
   RT_finalTest_restudyGroup = length(finalTest_restudyGroup_activationLog{pattern});
    
   % test network
   
    % let network settle until threshold
    finalTest_testGroup_activationLog{pattern} = memoryNet_testGroup.runTrialUntilThreshold(input, N_threshold);
    
    % compute accuracy & RT
    accuracy_finalTest_testGroup(pattern) = memoryNet_testGroup.computeAccuracy(input);
    RT_finalTest_testGroup = length(finalTest_testGroup_activationLog{pattern});
    
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
title('final test (restudy group)');

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
title({'final test (test group)'});

subplot(2,7,14);
imagesc(W_afterTestDecay);  
caxis([w_limit]);
title({'weights at final test', '(test group)'});
