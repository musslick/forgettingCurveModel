classdef simpleMemoryNet < handle
    
    properties(SetAccess = public)
        
        W; % weight matrix
        init; % initial activation value
        gain; % gain of activation function
        
        tau; % integration constant
        threshold; % response threshold
        maxTimeSteps; % maximum amount of simulation time steps
        
        activation_log; % log of activation for all units
        activation; % current activation
    end
    
    methods
        
        % constructor
        function this = simpleMemoryNet(W_arg, init_arg, threshold_arg)
            this.W = W_arg;
            this.init = init_arg;
            this.threshold = threshold_arg;
            
            if(length(this.init) ~= size(this.W))
                error('Dimension of initial activation vector does not match dimension of weight matrix.');
            end
            
            if(size(this.W,1) ~= size(this.W,2))
                error('Weight matrix must be a square matrix.');
            end
            
            % default parameters
            this.gain = 1;
            this.tau = 0.01;
            this.maxTimeSteps = 1000;
            
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
            
            activation_log = this.activation_log;
            
        end
        
        % compute activation at the next time step
        function newAct = computeNewActivation(this, input)
            
            if(size(input,1) == 1)
                input = transpose(input);
            end
            
            netInput = this.W * this.activation + input;
            newAct = this.activation + this.tau * (-this.activation + 1./(1+exp(-this.gain*netInput)));
            
        end
        
        % adjust weights
        function W = adjustWeights(this)
            
            W_delta = zeros(size(this.W));
            
            % compute weight adjustments over time
            for t = 1:size(this.activation_log,1)
                current_activation = this.activation_log(t,:);
                W_delta = W_delta + this.computeWeightAdjustment(current_activation);
            end
            
            % apply weight adjustment
            W = this.W + W_delta;
            this.W = W;
            
        end
        
        function  W_delta = computeWeightAdjustment(this, activation)
            
            % make sure activation vetor is column vector
            if(size(activation,1) == 1)
                activation = transpose(activation);
            end
            Nunits = length(activation);
            
            W_delta = (ones(Nunits) - eye(Nunits)) .* (transpose(activation) * activation);
        end
        
    end
    
end