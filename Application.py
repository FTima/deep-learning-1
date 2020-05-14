import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_util import * 

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Explore your dataset
    show_data_info(train_x_orig, test_x_orig, train_y, test_y)
    train_x, test_x = reshape_image(train_x_orig, test_x_orig)

    pred_train ,pred_test = predict_using_L_layer_nn(train_x,test_x,train_y, test_y)
    print_mislabeled_images(classes, test_x, test_y, pred_test)



def show_data_info(train_x_orig, test_x_orig, train_y, test_y):
    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

def reshape_image(train_x, test_x):
    # Reshape the training and test examples
    # The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
    test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    return train_x, test_x


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
       
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2, cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")

        cost = compute_cost(A2,Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1,cache1,"relu")

        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict_using_two_layer_nn(train_x, test_x, train_y):
    n_x = np.shape(train_x)[0]
    n_h = 7
    n_y = np.shape(train_y)[0]
    layers_dims = (n_x, n_h, n_y)

    parameters = two_layer_model(train_x, train_y, layers_dims = layers_dims, num_iterations = 2500, print_cost=True)

    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X,parameters)
        
        cost = compute_cost(AL,Y)
        
        grads = L_model_backward(AL,Y,caches)
        
        parameters = update_parameters(parameters,grads,learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict_using_L_layer_nn(train_x, test_x, train_y, test_y):
    n_x = np.shape(train_x)[0]
    #  4-layer model
    n_h_1 = 20
    n_h_2 = 7
    n_h_3 = 5
    n_y = np.shape(train_y)[0]
    layers_dims = (n_x, n_h_1,n_h_2,n_h_3, n_y)

    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    return pred_train ,pred_test
    

if __name__ == "__main__":
    main()

