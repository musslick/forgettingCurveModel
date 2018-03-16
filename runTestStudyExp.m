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
Npairs=10;
% number of units per pair
Nunits = Npairs*2;
% initial activation
initAct = zeros(Nunits, 1);
wt_init=0.01;%0.01;%higher initial weights -> more noise
gain_init=1;%makes influence of input stronger
tau_init=0.01;
wtnoise=1;%1
in=100;
% weights
initWeightScale = wt_init+rand(Nunits)*wt_init*wtnoise;
Ws = (ones(Nunits) - eye(Nunits)) .* initWeightScale;
initWeightScale = wt_init+rand(Nunits)*wt_init*wtnoise;
Wt = (ones(Nunits) - eye(Nunits)) .* initWeightScale;

% activation threshold
threshold = 0.4;

%% initial study 
%ALL items, both re-studied later and tested later
%creates 20 x 20 matrix with only 10 "study trials"
externalInput=zeros(Nunits,Nunits);
for i=1:Npairs
    externalInput(i,i)=in;
    externalInput(i+Npairs,i)=in;
end;

%strengthen corresponding unit via study
memoryNet_study = simpleMemoryNet(Ws,initAct,threshold,gain_init,tau_init);
activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
Ws2 = memoryNet_study.adjustWeights();%iterate over
memoryNet_study = simpleMemoryNet(Wt,initAct,threshold,gain_init,tau_init);
activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
Wt2 = memoryNet_study.adjustWeights();%iterate over
figure;subplot(221);imagesc(Ws);colorbar;subplot(222);imagesc(Wt);colorbar;
subplot(223);imagesc(Ws2);colorbar;subplot(224);imagesc(Wt2);colorbar;

% templ=zeros(Nunits,1);in=1;
% for i=1:Npairs
%     externalInput=templ;
%     externalInput(i)=in;
%     externalInput(i+Npairs)=in;
%     %strengthen corresponding unit via study
%     memoryNet_study = simpleMemoryNet(Ws,initAct,threshold,gain_init,tau_init);
%     activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
%     Ws2(i,:,:) = memoryNet_study.adjustWeights();%iterate over
%     memoryNet_study = simpleMemoryNet(Wt,initAct,threshold,gain_init,tau_init);
%     activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
%     Wt2(i,:,:) = memoryNet_study.adjustWeights();%iterate over
% end;
% Ws2=squeeze(mean(Ws2,1));Wt2=squeeze(mean(Wt2,1));

%% restudy 
memoryNet_study = simpleMemoryNet(Ws2,initAct,threshold,gain_init,tau_init);
activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
Ws3 = memoryNet_study.adjustWeights();%iterate over

% %study ALL items, both re-studied later and tested later
% for i=1:Npairs
%     externalInput=templ;
%     externalInput(i)=in;
%     externalInput(i+Npairs)=in;
%     %strengthen corresponding unit via study
%     memoryNet_study = simpleMemoryNet(Ws2,initAct,threshold,gain_init,tau_init);
%     activation_log_study = memoryNet_study.runTrialUntilThreshold(externalInput,Nunits);
%     Ws3(i,:,:) = memoryNet_study.adjustWeights();%iterate over
% end;
% Ws3=squeeze(mean(Ws3,1));

%% test
externalInput=zeros(Nunits,Nunits);in=1;
for i=1:Npairs
    externalInput(i,i)=in;
end;
%strengthen corresponding unit via study
memoryNet_test = simpleMemoryNet(Wt2,initAct,threshold,gain_init,tau_init);
activation_log_study = memoryNet_test.runTrialUntilThreshold(externalInput,Nunits);
Wt3 = memoryNet_test.adjustWeights();%iterate over

% %test
% for i=1:Npairs
%     externalInput=templ;
%     externalInput(i)=in;
%     %strengthen corresponding unit via study
%     memoryNet_test = simpleMemoryNet(Wt2,initAct,threshold,gain_init,tau_init);
%     activation_log_study = memoryNet_test.runTrialUntilThreshold(externalInput,Nunits);
%     [Wt3(i,:,:),fract] = memoryNet_test.adjustWeights();%iterate over
% end;
% Wt3=squeeze(mean(Wt3,1));

%% quick plot
figure;subplot(321);imagesc(Ws);colorbar;subplot(322);imagesc(Wt);colorbar;
subplot(323);imagesc(Ws2);colorbar;subplot(324);imagesc(Wt2);colorbar;
subplot(325);imagesc(Ws3);colorbar;subplot(326);imagesc(Wt3);colorbar;

%% plot - HAVEN'T CHANGED THIS
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





