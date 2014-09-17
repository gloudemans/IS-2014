%RbmStack
% This class implements a stack of restricted Boltzmann machines. It 
% provides methods to train the stack using a greedy layerwise
% approach, to perform inference using the stack as an autoencoder,
% and to unroll the stack to initialize a deep feedforward network
% implementing either an autoencoder or a classifier.
%
% The RbmStack provides the following features:
%
% * Allows specification of architectures having varying numbers of
%   layers with specified size and activation functions.
% * Optionally implements dropout of visible and/or hidden neurons
%   during training.
% * Means to specify the learning rate, momentum and dropout 
%   probabilities for each layer during each training epoch.
% * Event callback mechanism for reporting training progress
%   and performance metrics.

classdef RbmStack < handle
    
    properties
        
        % Number of samples per training batch
        iBatchSamples = 100;
        
        % Variance for random weight initialization
        rInitialWeightVariance = 0.100;   
        
        % Normalize the dropout gradient by the number of weight updates
        % rather than the number of training samples.
        bNormalizeDropoutGradient = false;

        % Layer properties
        % * oaLayer(k).iSize
        % * oaLayer(k).sActivationUp - upward activation   ('Logistic', 'Linear', 'HyperbolicTangent')
        % * oaLayer(k).sActivationDn - downward activation ('Logistic', 'Linear', 'HyperbolicTangent')
        % * oaLayer(k).raaW - weights
        oaLayer;
          
    end % properties
    
    methods
        
        %% RbmStack
        % Construct a new object with the specified configuration. The 
        % configuration object must be a structure array and its length
        % must be one greater than the number of machines in the stack. 
        % Only the iSize parameter is used for the final layer oaLayer(end).
        %
        % * oaLayer - specifies the network configuration.
        
        function o = RbmStack(oaLayer)

            % Copy the layer properties
            o.oaLayer = oaLayer;
              
            % For each weight layer...
            for iLayer = 1:length(o.oaLayer)-1
            
                % Initialize random weights while adding extra biases
                o.oaLayer(iLayer).raaW = randn( ...
                    o.oaLayer(iLayer+0).iSize+1, ...
                    o.oaLayer(iLayer+1).iSize+1) * o.rInitialWeightVariance;

                % Clear the biases
                o.oaLayer(iLayer).raaW(:,end) = 0;
                o.oaLayer(iLayer).raaW(end,:) = 0;
              
            end % for
            
        end % function
        
        %% TrainAutoencoder
        % Perform greedy pre-training of the network using the specified
        % training data and options.
        %
        % *raaX - specifies the network training samples
        % *oOptions - specifies training options
        % *oOptions.oaLayer(iLayer).raDropV(iEpoch)
        % *oOptions.oaLayer(iLayer).raDropH(iEpoch)
        % *oOptions.oaLayer(iLayer).raMomentum(iEpoch)
        % *oOptions.oaLayer(iLayer).raRate(iEpoch)
        % *oOptions.oaLayer(iLayer).baSample(iEpoch)

        function TrainAutoencoder(o, raaX, oOptions)
         
            % Count the number of training samples
            iSamples = size(raaX,1);
                       
            % For each layer...
            for iLayer = 1:length(o.oaLayer)-1
   
                % Create a delta array to retain momementum state
                raaDelta = zeros(size(o.oaLayer(iLayer).raaW));
                
                % Create a diff array to retain current update
                raaDiff = zeros(size(o.oaLayer(iLayer).raaW));
                
                % Create an array to retain the layer output for 
                % training the next layer
                raaY = zeros(iSamples, size(o.oaLayer(iLayer).raaW,2)-1);
                
                % Get short references to layer parameters
                sActivationUp = o.oaLayer(iLayer).sActivationUp;
                sActivationDn = o.oaLayer(iLayer).sActivationDn;
                
                % For each training epoch...
                for iEpoch = 1:oOptions.iEpochs

                    % Get short references to epoch parameters
                    rDropV    = oOptions.oaLayer(iLayer).raDropV(iEpoch);
                    rDropH    = oOptions.oaLayer(iLayer).raDropH(iEpoch);
                    rMomentum = oOptions.oaLayer(iLayer).raMomentum(iEpoch);
                    rRate     = oOptions.oaLayer(iLayer).raRate(iEpoch);
                    bSample   = oOptions.oaLayer(iLayer).baSample(iEpoch);

                    % Clear the sample index
                    iIndex = 0;
                    
                    % Clear error accumulators for this layer
                    rTotalSe = 0;
                    rTotalE  = 0;

                    % While training samples remain...
                    while(iIndex<iSamples)

                        % Compute an indexer
                        ia = iIndex+1:min(iIndex+o.iBatchSamples,iSamples);
                         
                        % Get a batch of inputs
                        raaV0 = raaX(ia,:);
                        
                        % If we need to drop visible units...
                        if(rDropV>0)
                        
                            % Compute a mask
                            baaV = logical(rand(size(raaV0))<rDropV);

                            % Clear dropped states
                            raaV0(baaV) = 0;
                            
                        end % if

                        % Advance the markov chain V0->H1
                        [raaH1d, raaH1s] = UpdateStates(o, sActivationUp, o.oaLayer(iLayer).raaW, raaV0, rDropV);

                        % If stochastic sampling is enabled...
                        if(bSample)

                            % Use sampled states
                            raaH1 = raaH1s;

                        else

                            % Use deterministic states
                            raaH1 = raaH1d;

                        end % end

                        % If we need to drop hidden units...
                        if(rDropH>0)
                            
                            % Compute a mask
                            baaH = logical(rand(size(raaH1))<rDropH);

                            % Clear dropped states
                            raaH1(baaH) = 0;
                            
                        end % if

                        % Advance the markov chain H1->V2
                        [raaV2       ] = UpdateStates(o, sActivationDn, o.oaLayer(iLayer).raaW', raaH1, rDropH);

                        % If we need to drop visible units...
                        if(rDropV>0)
                            
                            % Clear dropped states
                            raaV2(baaV) = 0;
                        
                        end % if

                        % Advance the markov chain V2->H3
                        [raaH3       ] = UpdateStates(o, sActivationUp, o.oaLayer(iLayer).raaW,  raaV2, rDropV);

                        % If we need to drop hidden units...
                        if(rDropH>0)
                            
                            % Clear dropped states
                            raaH3(baaH) = 0;
                        
                        end % if

                        % Scale factor to average this batch
                        rScale = 1/length(ia);
                        
                        % If normalizing the dropout gradient by the number
                        % of weight updates rather the number of batch
                        % samples.
                        if(o.bNormalizeDropoutGradient)
                            
                            % If no visible layer dropout...
                            if(~rDropV)
                                
                                % Construct a null dropout matrix
                                baaV = zeros(size(raaV0));
                                
                            end % if
                            
                           % If no hidden layer dropout...
                            if(~rDropH)
                                
                                % Construct a null dropout matrix 
                                baaH = zeros(size(raaH1));
                                
                            end % if      
                            
                            % Compute normalizer matrix
                            raaN = 1./(double(~baaV)'*(~baaH));

                            % Compute the average difference between positive phase 
                            % up(0,1) and negative phase up(2,3) correlations
                            raaDiff(1:end-1,1:end-1) = (raaV0'*raaH1-raaV2'*raaH3).*raaN;
                            
                        else
                            
                            % Scale all weights uniformly
                            raaDiff(1:end-1,1:end-1) = (raaV0'*raaH1-raaV2'*raaH3)*rScale; 
                            
                        end % if
                          
                        % Compute bias gradients
                        raDiffV = sum(raaV0-raaV2)*rScale;              
                        raDiffH = sum(raaH1-raaH3)*rScale;
                     
                        % Augment weight differences with biases
                        raaDiff(end,1:end-1) = raDiffH;
                        raaDiff(1:end-1,end) = raDiffV';

                        % Update the weight delta array using the current momentum and
                        % learning rate
                        raaDelta = raaDelta*rMomentum + raaDiff*rRate;

                        % Updated the weights
                        o.oaLayer(iLayer).raaW = o.oaLayer(iLayer).raaW + raaDelta;
                        
                        % If the visible layer used dropout...
                        if(rDropV)
                            
                            % Recompute H1 with no dropout
                            raaY(ia,:) = UpdateStates(o, sActivationUp, o.oaLayer(iLayer).raaW, raaX(ia,:), 0);
                            
                            % Recompute V2 based on the new H1
                            raaV2 = UpdateStates(o, sActivationDn, o.oaLayer(iLayer).raaW', raaY(ia,:), 0); 
                            
                        else
                            
                            % Use the prior computation
                            raaY(ia,:) = raaH1d;
                            
                            % If the hidden layer used dropout or sampling...
                            if(rDropH || bSample)
                                
                                % Recompute V2
                                raaV2 = UpdateStates(o, sActivationDn, o.oaLayer(iLayer).raaW', raaY(ia,:), 0); 
                            
                            end % if
                            
                        end % if
                        
                        % Gather error statistics for this minibatch
                        [rSe, rE] = GetErrors(o, raaX(ia,:), raaV2, sActivationDn);
                        
                        % Accumulate total errors
                        rTotalSe = rTotalSe+rSe;
                        rTotalE  = rTotalE + rE;
                        
                        % Advance to the next minibatch
                        iIndex = iIndex + length(ia);
                        
                    end % while
                    
                    % Finish the rmse calculation
                    rRmse = sqrt(rTotalSe/numel(raaX));
                    
                    % Record the error for this epoch
                    o.oaLayer(iLayer).raRmse(iEpoch) = rRmse; 
                    
                    % Finish rmse calculation
                    rError = rTotalE/numel(raaX);
                    
                    % Record the error for this epoch
                    o.oaLayer(iLayer).raError(iEpoch) = rError;

                    % Report training progress
                    oOptions.fEvent(iLayer, iEpoch, bSample, rDropV, rDropH, rRate, rMomentum, rRmse, rError);

                end % for
                
                % Current layer outputs are the next layer inputs
                raaX = raaY;
                    
            end % for

        end % function
                      
        %% UpdateStates
        % Given an activation type, a weight matrix, and a set of input 
        % layer states, return the logistic output layer states and
        % stochastic states.
        %
        % * sType - specifies the activation function
        % * raaW - specifies the weight matrix
        % * raaX - specifies the input states
        % * raaY - returns the output layer states
        % * baaY - returns the stochastic output states
        
        function [raaY, baaY] = UpdateStates(o, sType, raaW, raaX, rDropout)

            % If dropout was not specified...
            if(nargin<5)
                
                % Default to no dropout
                rDropout = 0;
                
            end % if
            
            % Compute the scale factor to compensate for dropout so that
            % average activations remain the same
            rScale = 1/(1-rDropout);
            
            % Compute activations
            raaA = [raaX*rScale ones(size(raaX,1),1)]*raaW(:,1:end-1);
                
            % Depending on the activation type...
            switch(sType)

                case 'Logistic'

                    % Compute the logistic function
                    raaY = 1./(1+exp(-raaA));

                case 'Linear'

                    % Compute output layer states
                    raaY = raaA;

                case 'HyperbolicTangent'

                    % Compute output layer states
                    raaY = tanh(raaA);

            end % switch
                                              
            % If stochastic binary states are required...
            if(nargout>1)

                % Depending on the activation type...
                switch(sType)

                    case 'Logistic'
                        
                        % Sample output layer states
                        baaY = raaY > rand(size(raaY));

                    case 'Linear'

                        % Sample output layer states
                        baaY = raaY + randn(size(raaY));

                    case 'HyperbolicTangent'

                        % Sample output layer states
                        baaY = raaY > 2*rand(size(raaY))-1;

                end % switch
                
            end % if
                 
        end % function
        
        %% Autoencode
        % For each specified sample, autoencode the sample as follows.
        % From the bottom to the top layer, compute the output distribution
        % for the machine given its input. From the top to the bottom
        % layer, Compute the input distribution for the layer given its
        % output. Return the resulting autoencoded sample.
        %
        % Note that the while loop batch processing below is a memory
        % optimization I needed simply because 32 bit matlab ran short of
        % memory when processing MNIST in one chunk. In general these
        % should not be ported for 64 bit use or even for normal 32 bit use
        % since they are ugly and the savings is modest.
        %
        % * raaX - specifies and returns patterns, one pattern per row
        
        function [raaX] = Autoencode(o, raaX)
            
            % Maximum number of patterns to process at once
            iBatch = 1000;
            
            % Measure the training data
            iSamples = size(raaX,1);
            
            % Clear the index
            iIndex = 0;
            
            % While training samples remain...
            while(iIndex<iSamples)

                % Compute an indexer
                ia = iIndex+1:min(iIndex+iBatch,iSamples);
                
                % Extract pattern batch
                raaB = raaX(ia,:);

                % For each layer in the network...
                for iLayer = 1:length(o.oaLayer)-1

                    % Propagate states upward
                    raaB = UpdateStates(o, o.oaLayer(iLayer).sActivationUp, o.oaLayer(iLayer).raaW, raaB);

                end % for 

                % For each layer in the network...
                for iLayer = length(o.oaLayer)-1:-1:1

                    % Propagate states downward
                    raaB = UpdateStates(o, o.oaLayer(iLayer).sActivationDn, o.oaLayer(iLayer).raaW', raaB);

                end % for
                
                % Save reconstruction states
                raaX(ia,:) = raaB;
                
                % Advance to the next batch
                iIndex = iIndex + length(ia);
                
            end % while
            
        end % function

        %% ComputeReconstructionError
        % Autoencode the specified samples and compute error metrics 
        % for the reconstructions.
        % 
        % * raaX - specifies the samples
        % * rRmse - returns the root mean squared error
        % * rError - returns the error measure for the output layer
        %   activation type
        
        function [rRmse, rError] = ComputeReconstructionError(o, raaX)
            
            % Process small batches to conserve memory
            iBatch = 1000;
            
            % Autoencode the specified samples to form reconstructions
            raaY = Autoencode(o, raaX);
            
            % Measure the data
            iSamples = size(raaX,1);
            
            % Clear the index
            iIndex = 0;
            
            % Clear the error accumulators
            rTotalE  = 0;
            rTotalSe = 0;
            
            % While training samples remain...
            while(iIndex<iSamples)
                
                % Compute an indexer
                ia = iIndex+1:min(iIndex+iBatch,iSamples);
                
                % Get errors for this batch
                [rSe, rE] = GetErrors(o, raaX(ia,:), raaY(ia,:), o.oaLayer(1).sActivationDn);
                
                % Accumulate the error totals
                rTotalSe = rTotalSe+rSe;
                rTotalE = rTotalE + rE;
    
                % Increment the index
                iIndex = iIndex + length(ia);
            
            end % while
            
            % Average error over all samples
            rError = rTotalE/numel(raaX);
            
            % Root mean square error over all samples
            rRmse  = sqrt(rTotalSe/numel(raaX));
            
        end % function
        
        %% GetErrors
        % Compute the total squared error and the total activation function 
        % specific error measure for all specified samples.
        %
        % * raaX - specifies the autoencoder input
        % * raaY - specifies the autoencoder output
        % * rSe - returns the sum of squared error for all input elements
        % * rE - returns the sum of activation function specific errors for
        %   all elements

        function [rSe, rE] = GetErrors(o, raaX, raaY, sActivation)
            
            % Small value to avoid log underflows
            rEps = 1e-80;         
            
            % Sum all squared errors
            rSe = sum((raaX(:)-raaY(:)).^2);
            
            % Depending on the activation function type
            switch(sActivation)

                case 'Logistic'

                    % Compute the average cross entropy error
                    rE = -(dot(raaX(:),log(raaY(:)+rEps)) + dot(1-raaX(:),log(1-raaY(:)+rEps)));

                case 'Linear'

                    % Compute the squared error
                    rE = rSe;

                case 'Softmax'

                    % Compute the average cross entropy error
                    rE = -dot(raaX(:),log(raaY(:)+rEps));

            end % switch
                
        end
        
        %% Unroll
        % Extract initialization parameters for the specified deep neural 
        % network type.
        %
        % * bAutoencoder - specifies whether to extract an autoencoder or
        %   a classifier (true-autoencoder, false-classifier)
        % * oaLayer - returns the network initialization parameters
        % * oaLayer(k).sActivation - activation type for layer k
        % * oaLayer(k).raaW - weights for layer k
        
        function [oaLayer] = Unroll(o, bAutoencoder)
            
            % Initialize the output layer index
            k = 0;
            
            % For each weight layer... 
            for iLayer = 1:length(o.oaLayer)-1
                
                % Increment the output layer index
                k = k + 1;
                
                % Get the weights while ignoring reverse biases
                oaLayer(k).raaW = o.oaLayer(iLayer).raaW(:,1:end-1);
                
                % Get the type
                oaLayer(k).sActivation = o.oaLayer(iLayer).sActivationUp;

            end % end
            
            % If extracting an autoencoder...
            if(bAutoencoder)
                
                % For each weight layer...
                for iLayer = length(o.oaLayer)-1:-1:1
                    
                    % Increment the output layer index
                    k = k + 1;

                    % Get the transposed weights while ignoring forward biases
                    oaLayer(k).raaW  = o.oaLayer(iLayer).raaW(1:end-1,:)';
                    
                    % Get the type
                    oaLayer(k).sActivation = o.oaLayer(iLayer).sActivationDn;
                
                end % for
                    
            end % if
                
        end % function
      
    end % methods 
    
end % classdef
