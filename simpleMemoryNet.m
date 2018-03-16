classdef simpleMemoryNet < handle
    
    properties(SetAccess = public)
        
        W; % weight matrix
        init; % initial activation value
        gain; % gain of activation function
        W_gain; %gain of weights
        
        tau; % integration constant
        threshold; % response threshold
        maxTimeSteps; % maximum amount of simulation time steps
        actFinish;
        
        activation_log; % log of activation for all units
        activation; % current activation
        
        inhibition;
    end
    
    methods
        
        % constructor
        function this = simpleMemoryNet(W_arg,init_arg,threshold_arg,gain_arg,tau_arg,inhib_arg)
            this.W = W_arg;
            this.init = init_arg;
            this.threshold = threshold_arg;
            this.tau=tau_arg;
            this.inhibition = inhib_arg;
            this.gain=gain_arg;
            
            if(length(this.init) ~= size(this.W))
                error('Dimension of initial activation vector does not match dimension of weight matrix.');
            end
            
            if(size(this.W,1) ~= size(this.W,2))
                error('Weight matrix must be a square matrix.');
            end
            
            % default parameters
            %JWA
            %this.gain = 1;
            %this.tau = 0.0015;%0.01
            this.maxTimeSteps = 1000;
            this.W_gain=0.0005;
        end
        
        
        % run trial
        function  activation_log = runTrialUntilThreshold(this, externalInput, N_threshold)
            
            this.activation_log = [];
            this.activation = this.init;
            
            % make sure activation vetor is column vector
            if(size(this.activation,1) == 1)
                this.activation = transpose(this.activation);
            end
            
            % activation loop
            for t = 1:this.maxTimeSteps
                
                % stop if desired number of neurons passed threshold
                if(sum(this.activation > this.threshold) >= N_threshold)
                    break
                end
                
                % compute network activation
                this.activation = this.computeNewActivation(externalInput);
                this.activation_log = [this.activation_log; transpose(this.activation)];
            end
            
            if(sum(this.activation > this.threshold) >= N_threshold)
                this.actFinish=1;
            else;this.actFinish=0;
            end;
            
            activation_log = this.activation_log;
        end
        
        % compute activation at the next time step
        function newAct = computeNewActivation(this, input)
            %JWA
            gain_adj=randn(1);%/5
            trial_gain = this.gain+gain_adj;
            if(size(input,1) == 1)
                input = transpose(input);
            end
            
            netInput = this.W * this.activation + input;
            newAct = this.activation + this.tau * (-this.activation + 1./(1+exp(-trial_gain*netInput)));
        end
        
        % adjust weights
        function [W,fract] = adjustWeights(this)
            
            %JWA
            %adjust weights if activation finished
            if this.actFinish 
                W_delta = zeros(size(this.W));
                
                % compute weight adjustments over time
                for t = 1:size(this.activation_log,1)
                    current_activation = this.activation_log(t,:);
                    W_delta = W_delta + this.computeWeightAdjustment(current_activation)*this.W_gain;
                end
                
                % apply weight adjustment
                W = this.W + W_delta;
                fract=W(1,2)/this.W(1,2);
                this.W = W;
            else;W = this.W;fract=1;
            end;
        end
        
        function  W_delta = computeWeightAdjustment(this, activation)
            
            % make sure activation vector is column vector
            if(size(activation,1) == 1)
                activation = transpose(activation);
            end
            Nunits = length(activation);
            
            %JWA
            W_delta = (ones(Nunits) - eye(Nunits)) .* (transpose(activation) * activation);
        end
        
        function W = weaken(this)
            %JWA
            W_delta = zeros(size(this.W));
            
            % compute weight adjustments over time
            for t = 1:size(this.activation_log,1)
                current_activation = this.activation_log(t,:);
                W_delta = W_delta + this.computeWeightAdjustment(current_activation)*this.W_gain;
            end
            
            % apply weight adjustment
            W = this.W-W_delta;
            this.W = W;
        end
        
        
    end
end