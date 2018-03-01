% try out BCM learning rule to model failure
% failure: activation below threshold (this should be in zone of destruction)
% try to model different weight strenghts as a function of initial encoding + noise
% model initial encoding phase with multiple pairs in the same pool of neurons
% think about adding noise to the settling process

% try exponential forgetting
% also include potential weights between units of other pairs into pool of neurons (context effect)
% BCM rule should push them apart from correc t pairs
% BCM rule should push them more apart from correct pairs for test vs. study phase
% this may lead to a different shape of the forgetting curve

% model justin hilbert's retrieval induced forgetting:
% assume overlap between apple and pear, both connect to fruit
% when retrieving fruit-apple, then fruit-pair gets weakened relative to control
% when re-studying fruit-pair then connection get's strengthened again but the connection between apple and pear may weaken


% initialize memory network

% number of units
Nunits = 2;

% initial activation
initAct = zeros(Nunits, 1);

% weights
initWeightScale = 0.01;
W = (ones(Nunits) - eye(Nunits)) * initWeightScale;

% activation threshold
threshold = 0.4;

% run study trial
externalInput_study = 1 * ones(Nunits, 1);
memoryNet_study = simpleMemoryNet(W, initAct, threshold);
activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput_study, Nunits);
W_study = memoryNet_study.adjustWeights();

% run test trial
externalInput_test = 1 * ones(Nunits, 1);
externalInput_test(2:end) = 0;
memoryNet_test = simpleMemoryNet(W, initAct, threshold);
activation_log_test= memoryNet_test.runTrialUntilThreshold(externalInput_test, Nunits);
W_test = memoryNet_study.adjustWeights();

% plot activations
all_activation = [activation_log_study(:); activation_log_test(:)];
subplot(2,2,1);
plot(activation_log_study);
ylim([min(all_activation) max(all_activation)]);
xlabel('time');
ylabel('unit activation');
title('study condition');
subplot(2,2,2);
plot(activation_log_test);
ylim([min(all_activation) max(all_activation)]);
xlabel('time');
ylabel('unit activation');
title('test condition');

% plot weight matrix
allWeights = [W_study(:); W_test(:)];
subplot(2,2,3);
imagesc(W_study);
colorbar;
caxis([min(allWeights), max(allWeights)]);
title('weights after training');
subplot(2,2,4);
imagesc(W_test);
colorbar;
title('weights after training');
caxis([min(allWeights), max(allWeights)]);


