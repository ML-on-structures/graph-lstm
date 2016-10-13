"""
This file builds a wrapper for using MLSL along with
a Neural Network sitting on top of it.
The neural network, for each of its entry,
takes in a feature vector defining inputs along with a
set of features outputting from an MLSL lying underneath
"""
from json_plus import Serializable
from mlsl import MLSL, softmax, get_objective_derivative
from dnn import DNN
import numpy as np

class MLSLNN(Serializable):
    """
    This class initializes a neural network
    based on the size of features per entry along
    with a provided MLSL which generates certain number of outputs
    """

    def __init__(self):
        pass

    def initialize(self, mlsl, nnl, seed=None, weight_range=1.0, outputs_from_mlsl=None, use_softmax=True):
        """
        Initialize an object of this class that binds a new NN on top
        of an existing MLSL object
        :param mlsl:
        :type mlsl: MLSL
        :param nnl:
        :type nnl: list
        :param seed:
        :type seed:
        :param weight_range:
        :type weight_range:
        :return:
        :rtype:
        """
        self.mlsl_output_size = mlsl.output_sizes[-1] if outputs_from_mlsl else outputs_from_mlsl

        # Change input size of Neural net to assigned feature size plus MLSL outputs
        nnl[0]+=self.mlsl_output_size

        self.outputs_from_mlsl = outputs_from_mlsl

        self.mlsl = mlsl
        self.nnet = DNN()
        self.nnet.initialize(nnl=nnl,seed=seed, weight_range=weight_range)
        self.use_softmax = use_softmax

    def forward(self, input_to_mlsl, additional_input_to_nn, target):
        """
        This runs a forward through the entire model comprising of an MLSL
        followed by a NN
        :param input_to_mlsl:
        :type input_to_mlsl:
        :param additional_input_to_nn:
        :type additional_input_to_nn:
        :return:
        :rtype:
        """
        mlsl_output = self.mlsl.forward_instance(input_to_mlsl, 0)
        input_to_nn = np.concatenate((mlsl_output[:self.mlsl_output_size], additional_input_to_nn))
        nnet_output = self.nnet.forward(input_to_nn)
        if self.use_softmax:
            nnet_output = softmax(nnet_output)

        return nnet_output

    def get_objective_derivative(self, output, target):
        if self.use_softmax:
            return output - target
        else:
            raise ValueError


    def backward(self, loss_deriv, instance_node):

        # Run derivative through LSTM first

        nn_deriv = self.nnet.backward_adadelta(loss_deriv)

        deriv = nn_deriv[:self.mlsl_output_size]

        self.mlsl.calculate_backward_gradients(instance_node, deriv, 0)
        self.mlsl.update_LSTM_weights(instance_node, 0)
        # updating the weights of the LSTM modules and
        # updating momentum_dW of LSTM modules with sums of dWs
        # and the other variables for adadelta
        # these momentum/adadelta specific updates happen regardless of whether we use steady rate, momentum, or adadelta
        # if we use steady rate those variables play no role in the computation of dW
        for d in range(self.mlsl.max_depth + 1):
            self.mlsl.lstm_stack[d].WLSTM += self.mlsl.sum_of_dWs[d] / self.mlsl.number_of_nodes_per_level[d]
            self.mlsl.lstm_stack[d].momentum_dW = self.mlsl.sum_of_dWs[d] / self.mlsl.number_of_nodes_per_level[d]
            self.mlsl.lstm_stack[d].tot_gradient_weight = self.mlsl.sum_tot_delta_weight[d] / self.mlsl.number_of_nodes_per_level[d]
            self.mlsl.lstm_stack[d].tot_sq_gradient = self.mlsl.sum_tot_sq_gradient[d] / self.mlsl.number_of_nodes_per_level[d]
            self.mlsl.lstm_stack[d].tot_delta_weight = self.mlsl.sum_tot_delta_weight[d] / self.mlsl.number_of_nodes_per_level[d]
            self.mlsl.lstm_stack[d].tot_sq_delta = self.mlsl.sum_tot_sq_delta[d] / self.mlsl.number_of_nodes_per_level[d]


    def run_through_the_model(self, instance_node, target, additional_input_to_nn):
        self.mlsl._reset_learning_parameters()
        return self.backward(self.get(self.forward(instance_node, additional_input_to_nn), target), instance_node)
