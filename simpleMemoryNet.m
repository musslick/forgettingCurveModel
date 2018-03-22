classdef simpleMemoryNet < handle
    
    properties(SetAccess = public)
        
        W; % weight matrix
        init; % initial activation value
        bias; % constant bias added to net input
        gain; % gain of activation function
        eta; % learning rate
        W_gain; %gain of weights
        
        learningRule;  % string that indicates learning rule, possible rules: 'Hebbian', 'BCM'
        
        tau; % integration constant
        threshold; % response threshold
        maxTimeSteps; % maximum amount of simulation time steps
        actFinish;
        
        activation_log; % log of activation for all units
        activation_softmax_log; % log of sotfmaxed activation for all units
        activation; % current activation
    end
    
    methods
        
        % constructor
        function this = simpleMemoryNet(W_arg,init_arg,threshold_arg,gain_arg,tau_arg,varargin)
            this.W = W_arg;
            this.init = init_arg;
            this.threshold = threshold_arg;
            this.tau=tau_arg;
            this.gain=gain_arg;
            this.bias = -2;
            this.eta = 0.1;
            
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
            
            if(length(varargin)>= 1) 
                this.learningRule = varargin{1};
            else
                this.learningRule = 'Hebbian';
            end
        end
        
        % run trial
        function  activation_log = runTrialUntilThreshold(this, externalInput, N_threshold)
            
            this.activation_log = [];
            this.activation_softmax_log = [];
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
            
            % softmax all activations
            activation_softmax = exp(this.activation)./sum(exp(this.activation));
            this.activation_softmax_log = [this.activation_softmax_log; transpose(activation_softmax)];
            
            if(sum(this.activation > this.threshold) >= N_threshold)
                this.actFinish=1;
            else
                warning('stopped integrating before reaching threshold.');
                this.actFinish=0;
            end
            
            activation_log = this.activation_log;
        end
        
        % compute activation at the next time step
        function newAct = computeNewActivation(this, input)
            %trial_gain = this.gain+randn(1);
            trial_gain = this.gain;
            if(size(input,1) == 1)
                input = transpose(input);
            end
            
            netInput = this.W * this.activation + input + this.bias;
            newAct = this.activation + this.tau * (-this.activation + 1./(1+exp(-trial_gain*netInput)));
        end
        
        % adjust weights
        function [W,fract] = adjustWeights(this)
            
            W_delta = zeros(size(this.W));
            
            % compute weight adjustments over time
            for t = 1:size(this.activation_log,1)
                if(strcmp(this.learningRule, 'Hebbian'))
                    
                W_delta = W_delta + this.HebbianWeightAdjustment(t); 
                
                elseif(strcmp(this.learningRule, 'BCM'))
                    
                    W_delta = W_delta + this.BCMWeightAdjustment(t); 
                    
                else
                    error(['Learning rule "' + this.learningRule + '" not implemented']);
                end
            end
            
            % apply weight adjustment
            W = this.W + W_delta;
            fract=W/this.W;
            this.W = W;
        end
        
        % weight adjustment based on BCM learning rule 
        function W_delta = BCMWeightAdjustment(this, t)
            
            y = this.activation_log(t,:);
            if(t <= 1)
                x = transpose(this.init);
            else
                x = this.activation_log(t-1,:);
            end
            
            % check for dimensions
            if(size(x,1) > 1)
                x = transpose(x);
            end
            
            Nunits = length(y);
            
            % compute spatial average
            theta_M = mean(y.^2);
            
            % compute weight adjustment
            W_delta = this.eta * (ones(Nunits) - eye(Nunits)) .* (transpose(y .* (y - theta_M)) * x);
            
            
        end
        
        % implements weight decay
        function [W] = decayWeights(this, decayRate, num_iterations, varargin)
            
            if(length(varargin) >= 1)
                noise = varargin{1};
            else
                noise = 0;
            end
            
            for t = num_iterations
                this.W = this.W - decayRate * this.W + randn(size(this.W))*noise;
            end
            
            W = this.W;
        end
        
        % weight adjustment based on Hebbian learning rule
        function  W_delta = HebbianWeightAdjustment(this, t)
            
            y = this.activation_log(t,:);
            if(t <= 1)
                x = this.init;
            else
                x = this.activation_log(t-1,:);
            end
            
            % make sure activation vector is column vector
            if(size(this.activation_log,1))
                y = transpose(y);
            end
            Nunits = length(y);
            
            %JWA
            W_delta = (ones(Nunits) - eye(Nunits)) .* (y*transpose(x));
            %currently just adding the same # to element in the matrix ... 
            %W_delta = (ones(Nunits) - eye(Nunits)) .* (transpose(activation) * activation);
        end
        
        function W = weaken(this)
            %JWA
            W_delta = zeros(size(this.W));
            
            % compute weight adjustments over time
            for t = 1:size(this.activation_log,1)
                current_activation = this.activation_log(t,:);
                W_delta = W_delta + this.HebbianWeightAdjustment(current_activation);%*this.W_gain;
            end
            
            % apply weight adjustment
            W = this.W-W_delta;
            this.W = W;
        end
        
        function accuracy = computeAccuracy(this, correctOutput)
            
            act = this.activation_log(end,:);
            act_thresh = zeros(size(act));
            act_thresh(act < this.threshold) = 0;
            act_thresh(act >= this.threshold) = 1;
            correctOutput(correctOutput <= 0) = 0;
            correctOutput(correctOutput > 0) = 1;
            
            accuracy = isequal(act_thresh, correctOutput);
            
        end
        
    end
end