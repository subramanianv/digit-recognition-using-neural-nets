function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

J=0;
zh=zeros(m,10);
delta2=0;
delta1=0;
for i=1:m
    yi=zeros(1,num_labels);
    yi(y(i))=1;
    xi=X(i,:);
   % Feed forward %
    z2=Theta1*xi' ;
    a2=sigmoid(z2);
    a2=[1;a2];
    z3=Theta2*a2;
    h=sigmoid(z3);
    a3=h;
    % End of Feed forward %
    s = (-yi) * log(h) - (1-yi)*log(1-h) ;
    J= J + s ;
    
    %backprop
    
    d3= a3 - yi';
    d2= (Theta2(:,2:end)' * d3 ) .* sigmoidGradient(z2) ;
    delta2 = delta2 + (d3*a2');
    delta1 = delta1 + (d2*xi);
    
    
end

J=J/m ;

s = trace(Theta1(:,2:end)* Theta1(:,2:end)') + trace(Theta2(:,2:end)* Theta2(:,2:end)');
J = J + ((s*lambda)/(2*m)) ;

Theta1_grad = (1/m) * delta1;
Theta2_grad = (1/m) * delta2;

Theta1_grad(:,2:end)= Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end));
Theta2_grad(:,2:end)= Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end));

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
