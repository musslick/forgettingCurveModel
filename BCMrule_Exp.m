clear all;
clc;

% experiment parameters

% number of pairs to be memorized
Npairs=10;

% network configuration

% network parameters
Nunits = 2*Npairs;      % number of units
N_threshold = 2;         % # of units required to reach threshold
w_init_scale = 0.01;    % scale of initial weights
gain = 1;                     % gain of activation function
tau = 0.1;                  % time integration constant
eta = 1;                  % BCM learning rate
threshold = 0.2;          % integration threshold (between 0 and 1)
inputStrength = 1;      % strength of input
maxTimeSteps = 1000; % maximum number of time steps

% initial activation
Act_init = zeros(Nunits, 1);

% generate weight matrix
W_init = rand(Nunits) * w_init_scale - w_init_scale/2;

% create memory network
memoryNet_study = simpleMemoryNet(W_init, Act_init, threshold, gain, tau, 'BCM');
memoryNet_study.maxTimeSteps = maxTimeSteps;
memoryNet_study.eta = eta;

%% initial study

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
    activation_log = memoryNet_study.runTrialUntilThreshold(input, N_threshold);
    
    % adjust weights
    [W,fract] = memoryNet_study.adjustWeights();
    
end

% log weights
W_afterStudy = memoryNet_study.W;

% store study net to save deep copy
save('studyNet_tmp.mat', 'memoryNet_study');

%% weight decay

%% restudy phase

% generate net for restudy condition
memoryNet_restudy = memoryNet_study;




%% plot
w_limit = [min([W(:); W_init(:)]) max([W(:); W_init(:)])];
figure(1);

subplot(1,2,1);
imagesc(W_init);  colorbar;
caxis([w_limit]);
title('weights before training');

subplot(1,2,2);
imagesc(W);  colorbar;
caxis([w_limit]);
title('weights after training');


