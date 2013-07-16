input_layer_size  = 784 ;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;
lambda_range=[3];
lambda_best=lambda_range(1);
%load('train_data.mat');
m = size(X, 1);
testCost=realmax;
for i=1:size(lambda_range,2)
    

   initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
   initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


   initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
   fprintf('\nTraining Neural Network... \n')

    options = optimset('MaxIter', 50);

    
    lambda = lambda_range(i)


    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    cost = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,Xtest,yTest,lambda);  
    if cost < testCost
        
        testCost=cost
        bestparams=nn_params;
        lambda_best=lambda
    end
 
end

fprintf('%f.\n',lambda_best);
fprintf('Program paused. Press enter to continue.\n');
pause;



Theta1 = reshape(bestparams(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(bestparams((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nVisualizing Neural Network... \n')

pred = predict1(Theta1, Theta2, Xtest);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yTest)) * 100);



