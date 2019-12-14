import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
%matplotlib inline

class neuralNetwork

	# initialise the neutral network
	def __init__( self, inputnodes, hiddennodes, outputnodes, learningrate )
		#self the initial nodes
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#learning rate
		self.lr = learningrate
		
		#we define the weights from w_i_j, where link is from node i to node j in the next layer
		self.wih = numpy.random.rand ( self.hnodes, self.inodes ) - 0.5
		self.who = numpy.random.rand ( self.onodes, self.hnodes ) - 0.5
		
		#optional advanced weight setup
		#causing error
		#self.wih = numpy.random.normal ( 0.0, pow( self.hnodes, -0.5 ), ( self.hnodes, self.inodes ) )
		#self.who = numpy.random.normal ( 0.0, pow ( self.onodes, -0.5 ), ( self.onodes, self.hnodes ) )
		
		# activation function is the sigmoid function
		#def self.activation_function
		#	return scipy.special.expit(x);
		self.activation_function = lambda x  scipy.special.expit( x )
		pass
		
	# train the neural network
	def train ( self, inputs_list, targets_list )
		# converts inputs list to 2d array
		inputs = numpy.array ( inputs_list, ndmin = 2 ).T
		targets = numpy.array ( targets_list, ndmin = 2 ).T
		
		# calculate signals into hidden layer
		hidden_inputs = numpy.dot ( self.wih, inputs )
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function( hidden_inputs )
		
		# calculate the signals emerging from final output layer
		final_inputs = numpy.dot ( self.who, hidden_outputs )
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function ( final_inputs )
		
		# error is the ( target - actual )
		output_errors = targets - final_outputs
		
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = numpy.dot ( self.who.T, output_errors )
		# update the weights for the links between the hidden and output layers
		self.who += self.lr  numpy.dot ( ( output_errors  final_outputs  ( 1.0 - final_outputs ) ), numpy.transpose ( hidden_outputs) )
		
		# update the weights for the links between the input and hidden layers
		self.wih += self.lr  numpy.dot ( ( hidden_errors  hidden_outputs  ( 1.0 - hidden_outputs ) ), numpy.transpose ( inputs ) )
		
		
		pass

	# query the neural network
	def query ( self, inputs_list )
	   # converts inputs list into to 2d array
	   inputs = numpy.array ( inputs_list, ndmin = 2 ).T	
	
	   # calculate the signals into hidden layer
	   hidden_inputs = numpy.dot ( self.wih, inputs )
	   # calculate the signals emergin from output layer
	   hidden_outputs = self.activation_function ( hidden_inputs )
	   
	   # calculate signals into final output layer
	   final_inputs = numpy.dot( self.who, hidden_outputs )
	   # calculate signals emerging from the final output layer
	   final_outputs = self.activation_function ( final_inputs )
	   
	   return final_outputs
	   

#we give initial values in variables
#nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork( input_nodes, hidden_nodes, output_nodes, learning_rate )

# load the mnist training data CSV file into a list
training_data_file = open ( mnist_datasetmnist_train.csv, 'r' )
training_data_list = training_data_file.readlines ()
training_data_file.close ()

# train the neural network

# go through all the records in the training data set
for record in training_data_list
    # split the record by the ',' commas
    all_values = record.split ( ',' )
    # scale and shift the inputs
    inputs = ( numpy.asfarray ( all_values [ 1 ] )  255.0  0.99 ) + 0.01
    # create the target output value ( all 0.01, except the desired label which is 0.99 )
    targets = numpy.zeros ( output_nodes ) + 0.01
    # all_values[ 0 ] is the target label for this record
    targets[ int ( all_values[ 0 ] ) ] = 0.99
    n.train ( inputs, targets )
    pass

# test the neural network

# load the mnist test data CSV file into a list
test_data_file  = open ( mnist_datasetmnist_test.csv, 'r' )
test_data_list = test_data_file.readlines ()
test_data_file.close ()

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for  record in test_data_list
    # split the record by the ',' commas
    all_values = record.split ( ',' )
    # correct answer is the first value
    correct_label = int ( all_values [ 0 ] )
    print ( correct_label, correct label )
    # scale and shift he inputs
    inputs = ( numpy.asfarray ( all_values [ 1 ] )  255.0  0.99 ) + 0.01
    # query the network
    outputs = n.query ( inputs )
    # the index of the highest value corresponds to the label
    label = numpy.argmax ( outputs )
    print ( label, networks's answer  )
    # append coorect or incorrect to list
    if ( label == correct_label )
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append ( 1 )
    else
            # add 0
            scorecard.append ( 0 )
            pass
    pass
            
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray ( scorecard )
print ( performance = , scorecard_array.sum()  scorecard_array.size )