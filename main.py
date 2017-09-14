import copy, numpy as np
# seeds random generator
np.random.seed(0)

# compute sigmoid nonlinearity
# Activation function - decides whether or not to accept the neuron, i.e, activate it.
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
# Derivative is the slope of a non-linear curve, i.e, our sigmoid.
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# training dataset generation
int2binary = {}
binary_dim = 8

# Setting highest no. to 256 so that we add atmost 8 digits(2^8)
largest_number = pow(2,binary_dim)
#converting numbers into binary form
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1  # learning rate
input_dim = 2  # input- we are adding 2 bits at a time
hidden_dim = 16 # hidden dimensions. Changing the value up or down affects the training differently 
output_dim = 1 # we get only one bit (0 or 1)


# initialize neural network weights
# SYsnapse represents the network lines (weights)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1  # 2 * 16 nets (2 * [0,1] - 1 == (-1,1)))
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # 16 * 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1	# 16 * 16

# initializing to zeroes
# fill weight update matrix with zeroes consisting of same dimensions as synapse_0,1,h
synapse_0_update = np.zeros_like(synapse_0) 
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
# 10000 traing iterations
# The loop ends at the last line of the code
for j in range(100000):
    
    # generate a simple addition problem (a + b = c)
    # diving by 2 so that (a [max 128] + b [max 128] = c [not greater than 256(our greatest no.)]) 
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    # answer in int (real number)
    c_int = a_int + b_int
    # answer in binary
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    # the NN prediction is stored here in future. so init to zeros
    d = np.zeros_like(c)

    overallError = 0
    
    # init
    layer_2_deltas = list()
    layer_1_values = list()

    # since there is no previous hidden layer, we are adding zeros to respective layers 
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        # x represents the 2 bits to be added
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        # y represents the real solution
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        # dot product between inputs (layer 0) and initial weights + dot product of current hidden layer and previous hidden layer weights
        # since a hidden layer is constructed based on present inputs and data from previous hidden layer
        # layer_1_values consists of values in form of an array in a list. layer_1_values[-1] == array (only),layer_1_values == list 
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        # output = dot of l1 and s1
        # 1*1 matrix is produced (1*8).(16*1)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        # error = binary difference of desired output and generated output
        layer_2_error = y - layer_2
        # layer 2 derivative. Consists of error diff at the output layer.
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        # layer_2_error[0] == array (i guess)
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        # layer_2[0][0] = exact value
        # d stores what the system predicted (value of a+b). It predicts ans as 0.33,etc.. We take it as 0.(0.63 ==1)
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        # hidden layer values (this becomes previous values in next iteration)
        layer_1_values.append(copy.deepcopy(layer_1))
    
    # init to zero
    future_layer_1_delta = np.zeros(hidden_dim)

	# let's backpropogate from layer_2 to layer_0    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])

        # The only current layer value is the last element (las neuron) in the hidden layer
        # rest all neurons iherit values from previous hidden layer (as shown in the figure)
        layer_1 = layer_1_values[-position-1]
        # leaving the last element. Starting from last but one
        prev_layer_1 = layer_1_values[-position-2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # multiplying and summing up the final update each iteration out of 100000 with learning rate
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    # re-initializing
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress after 1000 iterations
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

        

