#!/usr/bin/env python

# This class implements a neural net with forward and backpropagation.
# No specific loss function is used.  Rather, the backpropagation can
# backpropagate any derivative with respect to a loss function,
# and learn accordingly.

"""
Copyright (c) 2015, Camiolog Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""

# This code has been developed by Luca de Alfaro for Camiolog, Inc.,
# and is here released under BSD license.
# The code is derived from http://arctrix.com/nas/python/bpnn.py,
# developed by Neil Schemenauer <nas@arctrix.com> and placed in the
# public domain.


from json_plus import Serializable
import numpy as np
import unittest

# Type to be used for floats.
FLOAT_TYPE = 'double'

# Used for weight initialization.
NEURON_OVERLAP = 2.0

# These nets are between 0..1, hence the choice of sigmoid function.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# This is the derivative of a sigmoid as a function of the inputs.
def dsigmoid_in(x):
    e = np.exp(-x)
    return e / ((1.0 + e) ** 2)

# This is the derivative of a sigmoid as a function of the output.
def dsigmoid_out(y):
    return np.multiply(y, (1.0 - y))


class DNN(Serializable):
    """This class implements a neural net with inputs between 0..1,
    and outputs between 0..1. The net can have a specified number of
    neurons in the hidden layer."""

    def __init__(self, debug=False):
        """Do not call; no initialization is done.  Use the create method below."""
        self.debug = debug


    def initialize(self, nnl, seed=None, weight_range=1.0):
        """Produces a new net.
        nnl is a list, consisting of the number of values in each layer.
        The first element of nll is the number of inputs, and the last element is
        the number of outputs.
        Leaving the weight_range to None will cause it to be automatically chosen
        (recommended).
        Leaving the distribution to 'uniform' will use the uniform distribution,
        also recommended.
        seed is a seed for the random number generator.
        """
        # Sanity check.
        for n in nnl:
            assert n > 0
        self.nnl = nnl
        self.num_layers = len(nnl) - 1 # Number of layers.
        # Private random number generator.
        self.random_generator = np.random.RandomState(seed=seed)
        # w is a list of numpy arrays.
        # Each numpy array contains the matrix of weights from that layer to the next.
        # The element w[l][i,j] indicates the weight from element i of layer l
        # to element j of layer l + 1.
        self.w = []
        # c, old_dw are exactly like w, but stores weight momentums.
        # We set them to None initially, as we use them only if needed
        # according to the update method.
        self.c, self.old_dw = None, None
        # These are for AdaDelta
        self.tot_sq_delta, self.tot_sq_gradient = None, None
        self.tot_delta_weight, self.tot_gradient_weight = None, None
        for n in range(self.num_layers):
            # Initializes the weights.
            ww = np.matrix(self.random_generator.uniform(
                            -1.0, +1.0, size=(nnl[n] + 1, nnl[n + 1])))
            # Normalizes according to Nguyen-Widrow; see
            # http://web.stanford.edu/class/ee373b/nninitialization.pdf
            # Computes the modulus of the weights for each neuron.
            wwmod = np.sqrt(np.sum(np.multiply(ww, ww), axis=0))
            # Computes the ideal interval width, actually the reciprocal of the width.
            int_width = NEURON_OVERLAP * weight_range * np.power(nnl[n + 1], 1.0 / nnl[n])
            ww_norm = int_width * (ww / wwmod)
            self.w.append(ww_norm)
        # Creates the matrix b of activations.  Again, we store a list, in which b[n] is a
        # vector consisting of nnl[n] elements, containing the output activations of layer n.
        # Layer 0 consists of the inputs.
        # Note that the last element of self.b will be always set to 1, constituting the bias
        # for the activation potential of the net.
        self.b = []
        for n in range(self.num_layers + 1):
            self.b.append(np.matrix(np.ones(nnl[n] + 1, dtype=FLOAT_TYPE)))
        x, y = self.b[0].shape
        self.input_shape = (x, y - 1)


    def forward(self, bi):
        """Given a vector bi of nnl[0] values, computes the forward-propagation of the network,
        returning a vector bo consisting of nnl[-1] values, each between 0 and 1.
        This function also sets internally all the activations a and outputs b."""
        bii = np.matrix(bi)
        assert bii.shape == self.input_shape, "expected shape: %r actual shape: %r" % (self.input_shape, bii.shape)
        self.b[0][0, 0:self.nnl[0]] = bii
        # Propagates from layer n to layer n + 1.
        for n in range(self.num_layers):
            a = self.b[n] * self.w[n]
            self.b[n + 1][0, 0:self.nnl[n + 1]] = sigmoid(a)
        # The copy statement is necessary, or otherwise we can modify the b's
        # from the output, and potentially sabotage backpropagation.
        return self.b[self.num_layers][0, 0:self.nnl[self.num_layers]].copy()


    def backward(self, delta):
        """Implements backpropagation without updates.
        The input is a vector delta, of the same size of the
        outputs, giving \partial loss / \partial output.  The output is a vector, containing
        \partial loss / \partial input for every input, allowing the model to be chained.
        NOTE: this function must be called only after the forward step!"""
        return self._backward_update(delta, None)


    def backward_momentum_NM(self, delta, speed=0.5, N=0.5, M=0.3):
        """Implements backpropagation.  The input is a vector delta, of the same size of the
        outputs, giving \partial loss / \partial output.  The output is a vector, containing
        \partial loss / \partial input for every input, allowing the model to be chained.
        NOTE: this function must be called only after the forward step!"""
        # Defines the update function.
        def update_function(self, layer_idx, d, speed=speed, N=N, M=M):
            if self.c is None:
                self.c = [np.zeros((self.nnl[n] + 1, self.nnl[n + 1]), dtype=FLOAT_TYPE)
                          for n in range(self.num_layers)]
            # Update.
            wd = np.transpose(self.b[layer_idx]) * d
            self.w[layer_idx] -= speed * N * wd + M * self.c[layer_idx]
            self.c[layer_idx] = wd * speed
        return self._backward_update(delta, update_function)


    def backward_momentum(self, delta, speed=0.1, momentum=0.8):
        """Implements backpropagation.  The input is a vector delta, of the same size of the
        outputs, giving \partial loss / \partial output.  The output is a vector, containing
        \partial loss / \partial input for every input, allowing the model to be chained.
        NOTE: this function must be called only after the forward step!"""
        def update_function(self, layer_idx, d, speed=speed, momentum=momentum):
            if self.old_dw is None:
                self.old_dw = [np.zeros((self.nnl[n] + 1, self.nnl[n + 1]), dtype=FLOAT_TYPE)
                               for n in range(self.num_layers)]
            g = np.transpose(self.b[layer_idx]) * d
            dw = speed * g + momentum * self.old_dw[layer_idx]
            self.w[layer_idx] -= dw
            self.old_dw[layer_idx] = dw
        return self._backward_update(delta, update_function)


    def backward_adadelta(self, delta, learning_factor=1.0, epsilon = 0.1, decay=0.999):
        """This performs an adadelta update, see http://arxiv.org/abs/1212.5701 ,
        where learning_factor indicates how much we should learn from this particular example."""
        def update_function(self, layer_idx, d, epsilon=epsilon):
            if self.tot_sq_gradient is None:
                self.tot_sq_gradient = [0.0 for n in range(self.num_layers)]
                self.tot_sq_delta = [0.0 for n in range(self.num_layers)]
                self.tot_delta_weight = [0.0 for n in range(self.num_layers)]
                self.tot_gradient_weight = [0.0 for n in range(self.num_layers)]
            # Computes the gradient.
            g = np.transpose(self.b[layer_idx]) * d
            # Updates the gradient average.
            self.tot_sq_gradient[layer_idx] = self.tot_sq_gradient[layer_idx] * decay + np.sum(np.square(g))
            self.tot_gradient_weight[layer_idx] = self.tot_gradient_weight[layer_idx] * decay + g.size
            # Computes the speed.
            rms_delta = np.sqrt((self.tot_sq_gradient[layer_idx] + epsilon) /
                                (self.tot_gradient_weight[layer_idx] + epsilon))
            rms_gradient = np.sqrt((self.tot_sq_delta[layer_idx] + epsilon) /
                                   (self.tot_delta_weight[layer_idx] + epsilon))
            s = rms_delta / rms_gradient
            # Performs the update.
            dx = s * g
            self.w[layer_idx] -= dx * learning_factor
            # Updates the delta average.
            self.tot_sq_delta[layer_idx] = self.tot_sq_delta[layer_idx] * decay + np.sum(np.square(dx))
            self.tot_delta_weight[layer_idx] = self.tot_delta_weight[layer_idx] * decay + dx.size
        return self._backward_update(delta, update_function)


    def _backward_update(self, delta, update_function):
        """Implements backpropagation core.  The input is a vector delta, of the same size of the
        outputs, giving \partial loss / \partial output.  The output is a vector, containing
        \partial loss / \partial input for every input, allowing the model to be chained.
        Weights are updated if update is set to True.
        The function update_function is used to carry out the specific update.
        NOTE: this function must be called only after the forward step!"""
        # First, computes the derivatives wrt a[n], the activation layer.
        m = self.nnl[self.num_layers] # True number of outputs
        d = np.matrix(np.multiply(delta,
                        np.multiply(self.b[self.num_layers][0, 0:m], 1.0 - self.b[self.num_layers][0, 0:m])))
        # Then, iteratively for n going from the last layer to the first one:
        # - We update the weights leading from n to n + 1
        # - We compute d for the layer n.
        for n in range(self.num_layers - 1, -1, -1):
            m = self.nnl[n] # Number of true outputs at this level.
            # We do first the weight update, as it is very slightly faster.
            if update_function is not None:
                # Weight update.  change is \partial loss / \partial weight.
                # We use the full b, as change refers also to the activation potentials.
                update_function(self, n, d)
            # Computing d for previous layer. dd is \partial loss / \partial b
            # This should not include the activation potentials.
            dd = d * np.transpose(self.w[n][0:m, :])
            if n > 0:
                # Not the last layer.  We compute d as partial loss / \partial a
                d = np.multiply(dd, dsigmoid_out(self.b[n][0, 0:m]))
            else:
                # For the last layer, we just output d, since the inputs are equivalent
                # to b, and what we want is \partial loss / \partial input
                d = dd
        return d


class TestNet(unittest.TestCase):

    def test_backward(self):
        myrandom = np.random.RandomState(seed=0)
        net = DNN(debug=True)
        net.initialize([4, 2, 5, 3], 0)
        vi = myrandom.uniform(0.0, 1.0, 4)
        vo = net.forward(vi)
        # print "Output:", vo
        delta = myrandom.uniform(-1.0, 1.0, 3)
        d = net.backward_momentum_NM(delta)
        # print d

    def test_derivative(self):
        myrandom = np.random.RandomState(seed=0)
        net = DNN(debug=True)
        nnl = [3, 4, 2, 1]
        net.initialize(nnl, 0)
        # Backpropagates a dloss / dy of 1.
        bi = np.matrix(myrandom.uniform(0.0, 1.0, size=nnl[0]))
        y0 = net.forward(bi)
        # print "y0:", y0
        dd = net.backward(1.0)
        # Initializes the true derivatives to 0, as a placeholder.
        # Computes the true derivatives.
        epsilon = 0.001
        idx = 0
        bi[0, idx] += epsilon
        y1 = net.forward(bi)
        # print "y1:", y1
        # print "diff:", y1 - y0
        true_deriv = (y1 - y0) / epsilon
        # print "true deriv: ", true_deriv # [0, 0]
        # print "computed:   ", dd[0, idx]
        self.assertAlmostEqual(true_deriv[0, 0], dd[0, idx], 4)

    def test_update_NM(self):
        myrandom = np.random.RandomState(seed=0)
        net = DNN(debug=False)
        nnl = [4, 2, 3, 1]
        net.initialize(nnl, 0)
        # Backpropagates a dloss / dy of 1.
        bi = np.matrix(myrandom.uniform(0.0, 1.0, size=nnl[0]))
        y = []
        for i in range(10):
            y.append(net.forward(bi))
            net.backward_momentum_NM(1.0)
            if i > 0:
                self.assertLess(y[i], y[i - 1])
        # print "These must decrease (NM):"
        # print y

    def test_update_momentum(self):
        myrandom = np.random.RandomState(seed=0)
        net = DNN(debug=False)
        nnl = [4, 2, 3, 1]
        net.initialize(nnl, 0)
        # Backpropagates a dloss / dy of 1.
        bi = np.matrix(myrandom.uniform(0.0, 1.0, size=nnl[0]))
        y = []
        for i in range(10):
            y.append(net.forward(bi))
            net.backward_momentum(1.0)
            if i > 0:
                self.assertLess(y[i], y[i - 1])
        # print "These must decrease (momentum):"
        # print y

    def test_update_adadelta(self):
        myrandom = np.random.RandomState(seed=0)
        net = DNN(debug=False)
        nnl = [4, 2, 3, 1]
        net.initialize(nnl, 0)
        # Backpropagates a dloss / dy of 1.
        bi = np.matrix(myrandom.uniform(0.0, 1.0, size=nnl[0]))
        y = []
        for i in range(10):
            y.append(net.forward(bi))
            net.backward_adadelta(1.0)
            if i > 0:
                self.assertLess(y[i], y[i - 1])
        # print "These must decrease (adadelta):"
        # print y

class TestInit(unittest.TestCase):
    def test_start(self):
        net = DNN(debug=True)
        nnl = [4, 40, 1]
        net.initialize(nnl, 1)
        for i in range(5):
            fv = np.random.uniform(size=4)
            y = net.forward(fv)
            # print fv, y

class TestLearn(unittest.TestCase):

    def test_xor_momentum(self):
        import random
        N = 4000
        pats = [
            (np.array([0, 0]), 0),
            (np.array([0, 1]), 1),
            (np.array([1, 0]), 1),
            (np.array([1, 1]), 0),
        ]
        for k in range(20):
            net = DNN()
            net.initialize([2, 16, 1])
            e = np.zeros((N))
            for i in range(N):
                x, tgt = random.choice(pats)
                y = net.forward(x)
                dy = 2.0 * (y - tgt)
                e[i] = np.sum((y - tgt) ** 2)
                net.backward_momentum(dy)
                # print i, ":", e[i]
            avg_e = np.average(e[N/2:])
            print "MMT Avg error:", avg_e
            self.assertLess(avg_e, 0.01)

    def test_xor_adadelta(self):
        import random
        N = 4000
        pats = [
            (np.array([0, 0]), 0),
            (np.array([0, 1]), 1),
            (np.array([1, 0]), 1),
            (np.array([1, 1]), 0),
        ]
        for k in range(20):
            net = DNN()
            net.initialize([2, 16, 1])
            e = np.zeros((N))
            for i in range(N):
                x, tgt = random.choice(pats)
                y = net.forward(x)
                dy = 2.0 * (y - tgt)
                e[i] = np.sum((y - tgt) ** 2)
                net.backward_adadelta(dy)
                # print i, ":", e[i]
            avg_e = np.average(e[N/2:])
            print "ADA Avg error:", avg_e
            self.assertLess(avg_e, 0.01)

if __name__ == '__main__':
    unittest.main()

