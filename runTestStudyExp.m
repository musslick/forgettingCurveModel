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

rng('shuffle');reset(RandStream.getGlobalStream,sum(100*clock));
% initialize memory network
clear;
%number of pairs
Npairs=25;
% number of units per pair
Nunits = 2;
inhibition=0.1;
% initial activation
initAct = zeros(Nunits, 1);
wt_init=0.01;%higher initial weights -> more noise
gain_init=0.1;
tau_init=0.00157;
% weights
for i=1:Npairs;
    for j=1:Npairs;
        wt_adj=rand(1)*wt_init*5;%+wt_init;
        initWeightScale = wt_init+wt_adj(1);
        Ws(i,j,:,:) = (ones(Nunits) - eye(Nunits)) * initWeightScale;
        wt_adj=rand(1)*wt_init*5;%+wt_init;
        initWeightScale = wt_init+wt_adj(1);
        Wt(i,j,:,:) = (ones(Nunits) - eye(Nunits)) * initWeightScale;
    end;
    gains_s(i)=rand(1)+gain_init;
    gains_t(i)=rand(1)+gain_init;
    taus_s(i)=randn(1)*tau_init/10+tau_init;
    taus_t(i)=randn(1)*tau_init/10+tau_init;
end;

W_study=Ws;W_test=Wt;

% for later
%here we reconfigure weights to be the activation given the ROW of each
%corresponding column node. should be strongest for re-learned item
W_study2=squeeze(squeeze(Ws(:,:,2,1)));W_study3=W_study2;
W_test2=squeeze(squeeze(Wt(:,:,2,1)));W_test3=W_test2;

% activation threshold
threshold = 0.4;

%study ALL items, both re-studied later and tested later
externalInput_study = 1 * ones(Nunits, 1);
for i=1:Npairs
    %strengthen corresponding unit via study
    memoryNet_study = simpleMemoryNet(squeeze(Ws(i,i,:,:)),initAct,threshold,gains_s(i),taus_s(i),inhibition);
    activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput_study, Nunits);
    W_study(i,i,:,:) = memoryNet_study.adjustWeights();
    W_study2(i,i)=squeeze(W_study(i,i,2,1));
    memoryNet_test = simpleMemoryNet(squeeze(Wt(i,i,:,:)),initAct,threshold,gains_t(i),taus_t(i),inhibition);
    activation_log_study = memoryNet_test.runTrialUntilThreshold(externalInput_study, Nunits);
    W_test(i,i,:,:) = memoryNet_test.adjustWeights();
    W_test2(i,i)=squeeze(W_test(i,i,2,1));
end;
%at this point, W_study2 / W_test2 should be similar.

% run re-study trial
externalInput_study = 1 * ones(Nunits, 1);
for i=1:Npairs
    %strengthen corresponding unit via study
    memoryNet_study = simpleMemoryNet(squeeze(W_study(i,i,:,:)),initAct,threshold,gains_s(i),taus_s(i),inhibition);
    activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput_study, Nunits);
    study_a{i}.log=activation_log_study;
    W_study(i,i,:,:) = memoryNet_study.adjustWeights();
    W_study3(i,:)=squeeze(W_study(i,:,2,1));%1,1
end;
%figure;subplot(211);imagesc(W_study2);colorbar;
%subplot(212);imagesc(W_study3);colorbar;

% run test trial
externalInput_test = 1 * ones(Nunits, 1);
externalInput_test(2:end) = 0;
for i=1:Npairs
    memoryNet_test = simpleMemoryNet(squeeze(W_test(i,i,:,:)),initAct,threshold,gains_t(i),taus_t(i),inhibition);
    activation_log_test= memoryNet_test.runTrialUntilThreshold(externalInput_test, Nunits);
    test_a{i}.log=activation_log_test;
    [W_test(i,i,:,:),fract] = memoryNet_test.adjustWeights();
    %cycle through competitors and implement RIF
    for j=1:Npairs
        if i~=j
            %memoryNet_test = simpleMemoryNet(squeeze(W_test(i,j,:,:)),initAct,threshold,gains_t(i),taus_t(i),inhibition);
            %activation_log_test= memoryNet_test.runTrialUntilThreshold(externalInput_test, Nunits);
            %W_test(i,j,:,:) = memoryNet_test.weaken();
            W_test(i,j,:,:)=W_test(i,j,:,:)/((fract-1)/2+1);
        end;
    end;
    %adjust weights for both target and competitor
    W_test3(i,:)=squeeze(W_test(i,:,2,1));%2,1
end;

%W_study3(W_study3<0)=0;%zero out negative weights
%W_test3(W_test3<0)=0;%zero out negative weights

%% plot
%activations
all_activation = [activation_log_study(:); activation_log_test(:)];
yl=[min(all_activation) max(all_activation)];
figure;subplot(321);
for i=1:length(study_a);plot(study_a{1,i}.log);hold on;end;
ylim(yl);xlabel('time');ylabel('unit activation');title('study condition');
subplot(322);
for i=1:length(test_a);plot(test_a{1,i}.log);hold on;end;
ylim(yl);xlabel('time');ylabel('unit activation');title('test condition');
%  weight matrix
all_ws=[W_study2(:);W_test2(:);W_study3(:);W_test3(:)];
yl=[0, max(all_ws)];%yl=[min(all_ws), max(all_ws)];
subplot(323);imagesc(W_study2);colorbar;caxis(yl);
title('Post-learning study weights');xlabel('Input');ylabel('Output');
subplot(324);imagesc(W_test2);colorbar;caxis(yl);
title('Post-learning test weights');xlabel('Input');ylabel('Output');
subplot(325);imagesc(W_study3);colorbar;caxis(yl);
title('Post-restudy weights');xlabel('Input');ylabel('Output');
subplot(326);imagesc(W_test3);colorbar;caxis(yl);
title('Post-test weights');xlabel('Input');ylabel('Output');

%% implement forgetting over time!
%define memory as a thresholded signal:noise ratio
init_sn_study=zeros(Npairs,1);init_sn_test=init_sn_study
for i=1:Npairs
    %init_sn_study(i)=W_study3(i,i)/sum(W_study3(1:Npairs~=i,i));
    %init_sn_test(i)=W_test3(i,i)/sum(W_test3(1:Npairs~=i,i));
    init_sn_study(i)=W_study3(i,i)/sum(W_study3(:,i));
    init_sn_test(i)=W_test3(i,i)/sum(W_test3(:,i));
end;

allsn=[init_sn_study;init_sn_test];
snspace=linspace(min(allsn),max(allsn),10);
figure;h=histogram(init_sn_study,snspace,'FaceColor',[0 0 1]);hold on;
histogram(init_sn_test,snspace,'FaceColor',[1 0 0]);hold off;





