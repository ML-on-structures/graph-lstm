import lstm
import numpy as np
import random
from json_plus import Serializable, Storage
import unittest

class UnknownLearningMethod(Exception):
    def __init__(self, s):
        self.message = s


class MLSL(Serializable):

    def __init__(self, max_depth, output_sizes, node_feature_sizes,
                 learning_rate_vector, learning_method_vector,
                 shuffle_levels=[],
                 adadelta_parameters=None,
                 momentum_vector=None):
        """Initializes a multi-level LSTM.
        The ML-LSTM has max_depth layers.  Layer 0 is the root node.
        Layers max_depth - 1 to 0 have LSTMs in them.
        Layer max_depth is simply composed of graph nodes, which forward their
        features to the LSTMs of level max_depth - 1.
        The output of level i consists in the LSTM features computed from the children of i;
        it does not contain any features computed from the node at level i itself.
        The features of the node at level i will be passed to node at level i-1 along
        with the LSTM output.

        @param max_depth: As noted above.
        @param node_feature_sizes: How many features are produced by a node, according to its depth.
            This can go from 0 to max_depth (included).  Be careful: unless e.g. the graph is
            bipartite, you need to use the same number throughout.
        @param output_sizes: How many features are produced by LSTMs at different depth.  This
            does not need to be constant.
        @param learning_rate_vector: Vector of learning rates.
        @param learning_method_vector: Vector of learning methods. It can be None, in which case
            adadelta is used, or it can be a vector consisting of 'adadelta' or 'momentum'
            or 'steady_rate' (the latter is not recommended) for each layer.
        @param momentum_vector: vector containing momentums for learning.  It can be None if
            adadelta is used.
        @param adadelta_parameters: vector of adadelta parameters.  It can be None if momentum
            learning is used.
        @param shuffle_children: a list (or set) of depths at which shuffling is to occur.
        """
        # First, some sanity checks.
        assert max_depth > 0
        assert len(output_sizes) == max_depth
        assert len(node_feature_sizes) == max_depth + 1
        assert len(learning_method_vector) == max_depth
        assert adadelta_parameters is None or len(adadelta_parameters) == max_depth
        assert adadelta_parameters is not None or all(m != 'adadelta' for m in learning_method_vector)
        assert momentum_vector is None or len(momentum_vector) == max_depth
        assert momentum_vector is not None or all(m != 'steady_rate' for m in learning_method_vector)
        assert [i < max_depth for i in shuffle_levels]

        self.output_sizes = output_sizes
        self.node_feature_sizes = node_feature_sizes
        self.max_depth = max_depth
        self.learning_rate_vector = learning_rate_vector
        self.learning_method_vector = learning_method_vector
        self.adadelta_parameters = adadelta_parameters
        self.momentum_vector = momentum_vector
        self.shuffle_levels = shuffle_levels

        # Creates the list of LSTMs, one per level.
        self.lstm_stack = [lstm.LSTM() for _ in range(max_depth)]
        for l in range(max_depth):
            self.lstm_stack[l].initialize(
                node_feature_sizes[l + 1] + (0 if l == max_depth - 1 else output_sizes[l + 1]),
                output_sizes[l])

        # we need the following structures, when training with momentum and/or adadelta,
        # to keep track of the sum of dW at each level in order to update the momentum_dW
        # or the adadelta parameters of the respective LSTM modules.
        self.number_of_nodes_per_level = None
        self.sum_of_dWs = None
        self.sum_tot_sq_gradient = None
        self.sum_tot_gradient_weight = None
        self.sum_tot_sq_delta = None
        self.sum_tot_delta_weight = None


    def forward_propagation(self, instance_node, instance_depth=0):
        """Performs forward propagation through the multi-level LSTM structure.
         The node instance_node at depth instance_depth is propagated.
         The node should be an object of class InstanceNode."""
        # Prepares for back-propagation.
        self._reset_learning_parameters()
        input_sequence = np.array([])
        children_sequence = list(instance_node.get_children())
        if len(children_sequence) == 0:
            # FIXME We should really have a feature that describes the number of children.
            # This loses any data that might be associated with the node itself.
            return -100 * np.ones(self.output_sizes[instance_depth]) # no children signifier vector
        if instance_depth in self.shuffle_levels:
            # Shuffles children order if required.
            random.shuffle(children_sequence)
        for child_node in children_sequence:
            child_node_feature_vector = child_node.get_feature_vector()
            assert len(child_node_feature_vector) == self.node_feature_sizes[instance_depth + 1]
            # If we are not at the very bottom we need to get input from LSTM at the next level.
            LSTM_output_from_below = np.array([])
            if instance_depth < self.max_depth:
                 LSTM_output_from_below = self.forward_propagation(child_node, instance_depth=instance_depth + 1).reshape(
                     self.output_sizes[instance_depth + 1]) # recursive call
            # concatenate feature vector and input from LSTM output below
            full_feature_vector = np.concatenate((LSTM_output_from_below, child_node_feature_vector))
            # concatenate current feature vector to input sequence for the LSTM
            # TODO: This is very confusing; can you change this to use row and column stacking?
            input_sequence = np.concatenate((input_sequence, full_feature_vector))
        # forward the input sequence to this depth's LSTM
        input_sequence = input_sequence.reshape(len(children_sequence), 1, len(full_feature_vector))
        _, _, Y, cache = self.lstm_stack[instance_depth]._forward(input_sequence)
        # We store the state of the LSTM, so we can use it for back-propagation.
        instance_node.cache.lstm_cache = cache
        # we also need to save the sequence in the same order we used it.
        instance_node.children_sequence = children_sequence
        return Y


    def backward_propagation(self, instance_node, derivative, instance_depth=0):
        """Performs backward propagation, given a loss derivative for the outputs."""
        # First, we backpropagate through the layers the backward gradient.
        self._compute_backward_gradients(instance_node, derivative, instance_depth)
        # Second, we compute (but we do not apply) the update at all layers
        # of the MLSL.  We don't apply it because at every layer, there are in
        # general multiple instances of an LSTM, and we will have to add all the
        # updates for an LSTM at the same level before applying them.
        self._compute_LSTM_updates(instance_node, instance_depth)
        # Finally, once the updates have been computed, it applies them
        # to all the levels of the LSTM.
        self._apply_LSTM_updates()


    def _reset_learning_parameters(self):
        """This function should be called before any learning step."""
        self.number_of_nodes_per_level = [0 for _ in range(self.max_depth + 1)]
        self.sum_of_dWs = [0.0 for _ in range(self.max_depth)]
        self.sum_tot_sq_gradient =  [0.0 for _ in range(self.max_depth)]
        self.sum_tot_gradient_weight = [0.0 for _ in range(self.max_depth)]
        self.sum_tot_sq_delta = [0.0 for _ in range(self.max_depth)]
        self.sum_tot_delta_weight = [0.0 for _ in range(self.max_depth)]


    def _compute_backward_gradients(self, instance_node, derivative, instance_depth):
        """Recursive function to compute the backward gradients at all levels
        of the MLSL.  The gradients are left in instance_node.cache.weight_gradient."""
        dX, g, _, _ = self.lstm_stack[instance_depth].backward_return_vector_no_update(
            d = derivative, cache = instance_node.cache.lstm_cache)
        instance_node.cache.weight_gradient = g
        if instance_depth == self.max_depth:
            return
        for idx, item in enumerate(instance_node.children_sequence):
            if item.cache is None:
                continue
            input_derivatives = dX[idx, :, 0:self.output_sizes[instance_depth + 1]]
            if instance_depth < self.max_depth:
                feature_derivatives = dX[idx, :, self.output_sizes[instance_depth + 1]:]
            else:
                feature_derivatives = dX[idx, :, :]
            instance_node.children_sequence[idx].gradient = feature_derivatives
            self._compute_backward_gradients(item, input_derivatives, instance_depth + 1)


    def _compute_LSTM_updates(self, instance_node, current_depth):
        """Computes the update to the LSTM coefficients, recurrently down
        the tree of nodes."""
        # First, computes the update for the current node.
        method = self.learning_method_vector[current_depth]
        if method == "steady_rate":
            self._compute_update_LSTM_weights_steady_rate(instance_node, current_depth)
        elif method == "momentum":
            self._compute_update_LSTM_weights_with_momentum(instance_node, current_depth)
        elif method == "adadelta":
            self._compute_update_LSTM_weights_adadelta(instance_node, current_depth)
        else:
            raise UnknownLearningMethod(method)
        # Then, recurs down the tree.
        if current_depth == self.max_depth:
            return
        for item in instance_node.children_sequence:
            self._compute_LSTM_updates(item, current_depth + 1)


    def _compute_update_LSTM_weights_steady_rate(self, instance_node, current_depth):
        """Computes the LSTM weight update at steady rate."""
        if instance_node.cache is not None:
            dW = - self.learning_rate_vector[current_depth] * instance_node.cache.weight_gradient
            self.sum_of_dWs[current_depth] += dW
            self.number_of_nodes_per_level[current_depth] += 1


    def _compute_update_LSTM_weights_with_momentum(self, instance_node, current_depth):
        """Computes the LSTM weight update using momentum."""
        if instance_node.cache is not None:
            if self.lstm_stack[current_depth].momentum_dW is None: # initialize momentum of LSTM to zero
                self.lstm_stack[current_depth].momentum_dW = np.zeros(self.lstm_stack[current_depth].WLSTM.shape)
            dW = (- self.learning_rate_vector[current_depth] * instance_node.cache.weight_gradient
                  + self.momentum_vector[current_depth] * self.lstm_stack[current_depth].momentum_dW)
            self.lstm_stack[current_depth].WLSTM += dW
            self.sum_of_dWs[current_depth] += dW
            self.number_of_nodes_per_level[current_depth] += 1


    def _compute_update_LSTM_weights_adadelta(self, instance_node, current_depth):
        """Computes the LSTM weight update using adadelta."""
        # obtain adadelta parameters
        decay = self.adadelta_parameters[current_depth]["decay"]
        epsilon = self.adadelta_parameters[current_depth]["epsilon"]
        learning_factor = self.adadelta_parameters[current_depth]["learning_factor"]
        # do the adadelta updates
        if instance_node.cache is not None:
            instance_node.tot_sq_gradient = (self.lstm_stack[current_depth].tot_sq_gradient * decay
                                             + np.sum(np.square(instance_node.cache.weight_gradient)))
            instance_node.tot_gradient_weight = self.lstm_stack[current_depth].tot_gradient_weight * decay + 1.0
            # Computes the speed.
            rms_delta = np.sqrt((self.lstm_stack[current_depth].tot_sq_delta + epsilon)
                                / (self.lstm_stack[current_depth].tot_delta_weight + epsilon))
            rms_gradient = np.sqrt((instance_node.tot_sq_gradient + epsilon)
                                   / (instance_node.tot_gradient_weight + epsilon))
            s = rms_delta / rms_gradient
            # Computes the delta.
            delta = s * instance_node.cache.weight_gradient
            instance_node.tot_sq_delta = self.lstm_stack[current_depth].tot_sq_delta * decay + np.sum(np.square(delta))
            instance_node.tot_delta_weight = self.lstm_stack[current_depth].tot_delta_weight * decay + 1.0
            # Finally, updates the weights.
            dW = - delta * learning_factor
            self.sum_of_dWs[current_depth] += dW
            self.number_of_nodes_per_level[current_depth] += 1
            self.sum_tot_sq_gradient[current_depth] += instance_node.tot_sq_gradient
            self.sum_tot_gradient_weight[current_depth] += instance_node.tot_gradient_weight
            self.sum_tot_sq_delta[current_depth] += instance_node.tot_sq_delta
            self.sum_tot_delta_weight[current_depth] += instance_node.tot_delta_weight


    def _apply_LSTM_updates(self):
        """Applies the updates that have been computed to the LSTM."""
        for d in range(self.max_depth):
            self.lstm_stack[d].WLSTM += self.sum_of_dWs[d] / self.number_of_nodes_per_level[d]
            self.lstm_stack[d].momentum_dW = self.sum_of_dWs[d] / self.number_of_nodes_per_level[d]
            self.lstm_stack[d].tot_gradient_weight = self.sum_tot_delta_weight[d] / self.number_of_nodes_per_level[d]
            self.lstm_stack[d].tot_sq_gradient = self.sum_tot_sq_gradient[d] / self.number_of_nodes_per_level[d]
            self.lstm_stack[d].tot_delta_weight = self.sum_tot_delta_weight[d] / self.number_of_nodes_per_level[d]
            self.lstm_stack[d].tot_sq_delta = self.sum_tot_sq_delta[d] / self.number_of_nodes_per_level[d]



# the following class represents nodes of the unfoldings
# the MLSL module understands and can train and test on tree instances that are encoded as objects of this class

class InstanceNode(Serializable):
    """In order to use an MLSL, we need to pass to it a tree (tree, NOT dag)
    of these InstanceNode.
    At the end of the processing, the gradient attribute of each node
    will contain the backpropagation of the loss derivative to the feature
    vector of the node itself."""
    def __init__(self, feature_vector = None, label = None, id = None):
        self.id = id
        self.feature_vector = feature_vector
        self.label = label
        self.children = []
        self.children_sequence = [] # Stores the specific order by which the items were fed into the LSTM to update weights correctly
        # The gradient backpropagated at this node will be left here.
        # It can be used for further back-propagation as needed.
        self.gradient = None
        # Here we store intermediate values useful for the processing.
        self.cache = Storage()

    def set_label(self, label):
        self.label = label

    def get_number_of_children(self):
        return len(self.children)

    def get_label(self):
        return self.label

    def get_children(self):
        return self.children

    def get_feature_vector(self):
        return self.feature_vector


class SimpleLearningTest(unittest.TestCase):

    # FIXME: Move this to a unit test?
    def test_model(self, test_set):
        guesses = 0
        hits = 0
        found = {}
        missed = {}
        misclassified = {}
        for item in test_set:
            Y = self.forward_propagation(item)
            if Y is None:
                continue
            print Y
            predicted_label = Y.argmax()
            real_label = item.get_label()
            print "Predicted label ", predicted_label , " real label", real_label
            guesses += 1
            hits += 1 if predicted_label == real_label else 0
            if predicted_label == real_label:
                if real_label not in found:
                    found[real_label] = 1
                else:
                    found[real_label] += 1
            if predicted_label != real_label:
                if real_label not in missed:
                    missed[real_label] = 1
                else:
                    missed[real_label] += 1
                if predicted_label not in misclassified:
                    misclassified[predicted_label] = 1
                else:
                    misclassified[predicted_label] += 1
        print "LSTM results"
        print "============================================================="
        print "Predicted correctly ", hits , "over ", guesses, " instances."
        recall_list = []
        recall_dict = {}
        precision_dict = {}
        found_labels = set(found.keys())
        missed_labels = set(missed.keys())
        all_labels = found_labels.union(missed_labels)
        for label in all_labels:
            no_of_finds = float((0 if label not in found else found[label]))
            no_of_missed = float((0 if label not in missed else missed[label]))
            no_of_misclassified = float((0 if label not in misclassified else misclassified[label]))
            recall =  no_of_finds / (no_of_finds + no_of_missed)
            precision = no_of_finds / (no_of_finds + no_of_misclassified)
            recall_dict[label] = recall
            precision_dict[label] = precision
            recall_list.append(recall)
        avg_recall = np.mean(recall_list)
        print "Average recall ", np.mean(recall_list)
        if len(all_labels) == 2: # compute F-1 score for binary classification
            for label in all_labels:
                print "F-1 score for label ", label, " is : ",
                print 2 * (precision_dict[label] * recall_dict[label]) / (precision_dict[label] + recall_dict[label])
        return avg_recall
