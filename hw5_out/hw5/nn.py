import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    lim_h = np.sqrt(6)/np.sqrt(in_size+out_size)
    lim_l = - np.sqrt(6)/np.sqrt(in_size+out_size)

    W = np.random.uniform(lim_l,lim_h,(in_size,out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b
############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1/(1+np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    ##########################
    ##### your code here #####
    ##########################
    pre_act = (X @ W) + b
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)
    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    c = np.max(x,axis=1)
    c = c.reshape(-1,1)
    num = -np.exp(x-c)
    den = np.sum(num,axis=1)
    den = den.reshape(-1,1)
    res = num/den

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    g = np.argmax(probs,axis=1)
    p = np.argmax(y,axis=1)
    C = y.shape[0]

    acc = np.sum(g==p)/C
    loss = -np.sum(y*np.log(probs))

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################
    
    deriv = activation_deriv(post_act)
    grad_W = X.T @ (delta*deriv)
    n = (delta*deriv).shape[0]
    grad_b =  (np.ones((1,n)) @ (delta*deriv)).reshape(-1)
    grad_X = (delta*deriv) @ W.T
   
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    #########################
    #### your code here #####
    #########################
    x1 = x.shape[0]
    x2 = x.shape[1]
    num_batch = int(x1/batch_size)
    i = np.random.choice(x1,(num_batch,batch_size))
    #splitting
    # bx = np.split(x[i],num_batch)
    # by = np.split(y[i],num_batch)

    for j in range(len(i)):
        bx = x[i[j],:]
        by = y[i[j],:]
        batches.append((bx, by))



    return batches
