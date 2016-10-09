"""
Authors: Luca de Alfaro
This code is derived from a body of code
From https://gist.github.com/karpathy/587454dc0146a6ae21fc (MIT license)
This is a batched LSTM forward and backward pass
Modified by Luca de Alfaro, 2015.
"""

import numpy as np
from json_plus import Serializable
import unittest

class LSTM(Serializable):
    """Class implementing an LSTM."""

    def __init__(self):
        """We need an empty initializer, to be compatible with the Serializable
        interface."""
        pass

    def initialize(self, input_size, hidden_size, fancy_forget_bias_init=3):
        """
        Initialize parameters of the LSTM (both weights and biases in one matrix)
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        In the matrix there are inputs for:
        - 1 (bias)
        - Input
        - Hidden
        In the other dimension, there are four outputs, for:
        - Input to cell
        - Forget
        - Output
        - Gate
        """
        # +1 for the biases, which will be the first row of self.WLSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
        # self.WLSTM[0, :] = 0  # initialize biases to zero
        if fancy_forget_bias_init != 0:
            # forget gates get little bit negative bias initially to encourage them to be turned off
            # remember that due to Xavier initialization above, the raw output activations from gates before
            # nonlinearity are zero mean and on order of standard deviation ~1
            self.WLSTM[0, hidden_size:2 * hidden_size] = fancy_forget_bias_init

        # Init parameters for momentum update method.
        self.momentum_dW = None # Delta weights

        # Init parameters for ADADELTA update method.
        self.tot_gradient_weight, self.tot_delta_weight = 0, 0
        self.tot_sq_gradient, self.tot_sq_delta = 0, 0


    def clone(self):
        replica = LSTM()
        replica.input_size = self.input_size
        replica.hidden_size = self.hidden_size
        replica.WLSTM = np.copy(self.WLSTM)
        replica.momentum_dW = np.copy(self.momentum_dW)
        replica.tot_gradient_weight = self.tot_gradient_weight
        replica.tot_delta_weight = self.tot_delta_weight
        replica.tot_sq_gradient = self.tot_sq_gradient
        replica.tot_sq_delta = self.tot_sq_delta
        return replica


    def _forward(self, X, c0=None, h0=None):
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """
        n, b, isz = X.shape
        d = self.hidden_size
        if c0 is None: c0 = np.zeros((b, d))
        if h0 is None: h0 = np.zeros((b, d))
        assert(isz == self.input_size)

        # Perform the LSTM forward pass with X as the input
        m = self.WLSTM.shape[0]  # size of x plus h plus bias
        Hin = np.zeros((n, b, m))  # input [1, xt, ht-1] to each tick of the LSTM
        Hout = np.zeros((n, b, d))  # hidden representation of the LSTM (gated cell content)
        IFOG = np.zeros((n, b, d * 4))  # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4))  # after nonlinearity
        C = np.zeros((n, b, d))  # cell content
        Ct = np.zeros((n, b, d))  # tanh of cell content
        for t in xrange(n):
            # concat [x,h] as input to the LSTM
            prevh = Hout[t - 1] if t > 0 else h0 # previous cell output.
            # assembles cell input.
            Hin[t, :, 0] = 1  # bias
            Hin[t, :, 1:self.input_size + 1] = X[t]
            Hin[t, :, self.input_size + 1:] = prevh
            # print "Hin[%d]:\n" % t, Hin[t, :, :], '\n-------------------------------\n'
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(self.WLSTM)
            # non-linearities
            IFOGf[t, :, :3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * d]))  # sigmoids; these are the gates
            IFOGf[t, :, 3 * d:] = np.tanh(IFOG[t, :, 3 * d:])  # tanh
            # compute the cell activation
            prevc = C[t - 1] if t > 0 else c0
            # input * gate + forget * previous_cell; (2)
            C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d:] + IFOGf[t, :, d:2 * d] * prevc
            Ct[t] = np.tanh(C[t]) # nonlinearity
            Hout[t] = IFOGf[t, :, 2 * d:3 * d] * Ct[t] # output * cell (1)

        cache = {}
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        cache['h0'] = h0
        cache['n'] = n
        cache['b'] = b

        # We remember the cached values, so we don't need to plug them back in each time.
        self.cache = cache

        # return C[t], as well so we can continue LSTM with prev state init if needed
        return Hout, C[t], Hout[t], cache


    def clean_before_serialization(self):
        self.cache.clear()


    def _backward(self, dHout_in, cache=None, dcn=None, dhn=None):
        """Backward propagation through the LSTM.  dHout_in must have the same shape as Hout."""
        if cache is None:
            cache = self.cache
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        IFOG = cache['IFOG']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        c0 = cache['c0']
        h0 = cache['h0']
        n = cache['n']
        b = cache['b']
        d = self.hidden_size

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(self.WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n, b, self.input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        dHout = dHout_in.copy()  # make a copy so we don't have any funny side effects
        if dcn is not None: dC[n - 1] += dcn.copy()  # carry over gradients from later
        if dhn is not None: dHout[n - 1] += dhn.copy()
        for t in reversed(xrange(n)):

            tanhCt = Ct[t]
            # backpropagation through (1) for output
            dIFOGf[t, :, 2 * d:3 * d] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop.
            # this is the backprop of output on cells, (1) cont.
            dC[t] += (1 - tanhCt ** 2) * (IFOGf[t, :, 2 * d:3 * d] * dHout[t])

            if t > 0:
                dIFOGf[t, :, d:2 * d] = C[t - 1] * dC[t] # delta forget (through (2))
                dC[t - 1] += IFOGf[t, :, d:2 * d] * dC[t]
            else:
                dIFOGf[t, :, d:2 * d] = c0 * dC[t]
                dc0 = IFOGf[t, :, d:2 * d] * dC[t]
            # this completes backpropagation to the cell memory.

            # this is the gate * input portion, effects on gate and input.
            dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d:] * dC[t] # backprop of input through gate, part of (2)
            dIFOGf[t, :, 3 * d:] = IFOGf[t, :, :d] * dC[t] # backprop of gate through input, part of (2)

            # backprop activation functions
            dIFOG[t, :, 3 * d:] = (1 - IFOGf[t, :, 3 * d:] ** 2) * dIFOGf[t, :, 3 * d:]
            y = IFOGf[t, :, :3 * d]
            dIFOG[t, :, :3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * d]

            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(self.WLSTM.transpose())

            # backprop the identity transforms into Hin
            dX[t] = dHin[t, :, 1:self.input_size + 1]
            if t > 0:
                dHout[t - 1, :] += dHin[t, :, self.input_size + 1:]
            else:
                dh0 += dHin[t, :, self.input_size + 1:]

        return dX, dWLSTM, dc0, dh0


    def forward(self, X):
        """Forward function.  Can be called to predict outputs, and as preparation to backpropagation.
        X should be of shape (n, b, input_size), where n = length of sequence, b = batch size.
        If b = 1, one can also give dimensions (n, input_size) to X.
        """
        XX = X if X.ndim == 3 else X.reshape((X.shape[0], 1, X.shape[1]))
        _, _, o, _ = self._forward(XX)
        return o if X.ndim == 3 else o.flatten()


    def _adapt_input_derivative(self, d):
        """In an LSTM, we often have feedback only on the last result, only once
        all the sequence has been read.  This function takes d as given, and
        produces the internal representation that is needed.  The rule is as follows:
        - If d has dimension 3, then it is assumed that it comes already in the
          correct format.
        - If d has dimension 2, then it is assumed that there are batches, and
          that the data includes only the latest temporal step.  The previous
          temporal steps are filled in with zeros, as appropriate.
        - If d has dimension 1, then it is assumed that no batches are present,
          and that d refers only to the last temporal step.
          The other temporal steps are filled with zeros as required, and an
          appropriate array is returned.
        """
        if d.ndim == 3:
            return d
        elif d.ndim == 1:
            n = self.cache['n'] # N. of temporal steps
            assert(self.cache['b'] == 1)
            assert(d.size == self.hidden_size)
            dd = np.vstack((np.zeros((n - 1, self.hidden_size)), d))
            return dd.reshape(n, 1, self.hidden_size)
        elif d.ndim == 2:
            n = self.cache['n'] # N. of temporal steps
            batch_size, hidden_size = d.shape
            assert(batch_size == self.cache['b'])
            assert(hidden_size == self.hidden_size)
            other_times = np.zeros((n - 1, batch_size, hidden_size))
            return np.vstack((other_times, d.reshape(1, d.shape[0], d.shape[1])))


    def backward(self, d):
        """Backward function without learning.  Input is de loss / de output."""
        dd = self._adapt_input_derivative(d)
        _, _, _, dh0 = self._backward(dd)
        return dh0

    """ no update, and cache can be passed as parameter"""
    def backward_return_vector_no_update(self, d, cache):
        """Backward function without learning.  Input is de loss / de output."""
        self.cache = cache
        dd = self._adapt_input_derivative(d)
        dX, g, dc0, dh0 = self._backward(dd, cache = cache)
        return dX, g, dc0, dh0


    def backward_momentum(self, d, speed=0.1, momentum=0.8):
        """Implements backpropagation with momentum."""
        dd = self._adapt_input_derivative(d)
        _, g, _, dh0 = self._backward(dd)
        if self.momentum_dW is None:
            self.momentum_dW = np.zeros(self.WLSTM.shape)
        dW = - speed * g + momentum * self.momentum_dW
        self.momentum_dW = dW
        self.WLSTM += dW
        return dh0


    def backward_momentum_vector(self, d, speed=0.0001, momentum=0.0008):
        """Implements backpropagation with momentum."""
        dd = self._adapt_input_derivative(d)
        dX, g, dc0, dh0 = self._backward(dd)
        if self.momentum_dW is None:
            self.momentum_dW = np.zeros(self.WLSTM.shape)
        dW = - speed * g + momentum * self.momentum_dW
        self.momentum_dW = dW
        self.WLSTM += dW
        return dX, g, dc0, dh0


    def backward_adadelta(self, d, learning_factor=1.0, epsilon=0.001, decay=0.95):
        """Implements backpropagation with the ADADELTA method, see
        http://arxiv.org/abs/1212.5701
        learning_factor indicates how much we should learn from this particular example."""
        dd = self._adapt_input_derivative(d)
        _, g, _, dh0 = self._backward(dd)
        # Updates the gradient average.
        self.tot_sq_gradient = self.tot_sq_gradient * decay + np.sum(np.square(g))
        self.tot_gradient_weight = self.tot_gradient_weight * decay + 1.0
        # Computes the speed.
        rms_delta = np.sqrt((self.tot_sq_delta + epsilon) / (self.tot_delta_weight + epsilon))
        rms_gradient = np.sqrt((self.tot_sq_gradient + epsilon) / (self.tot_gradient_weight + epsilon))
        s = rms_delta / rms_gradient
        # Computes the delta.
        delta = s * g
        # Updates the delta average.
        self.tot_sq_delta = self.tot_sq_delta * decay + np.sum(np.square(delta))
        self.tot_delta_weight = self.tot_delta_weight * decay + 1.0
        # Finally, updates the weights.
        self.WLSTM -= delta * learning_factor
        return dh0

    def backward_adadelta_vector(self, d, learning_factor = 1.0, epsilon = 0.0001, decay = 0.95):
        """Implements backpropagation with the ADADELTA method, see
        http://arxiv.org/abs/1212.5701
        learning_factor indicates how much we should learn from this particular example."""
        dd = self._adapt_input_derivative(d)
        dX, g, dc0, dh0 = self._backward(dd)
        # Updates the gradient average.
        self.tot_sq_gradient = self.tot_sq_gradient * decay + np.sum(np.square(g))
        self.tot_gradient_weight = self.tot_gradient_weight * decay + 1.0
        # Computes the speed.
        rms_delta = np.sqrt((self.tot_sq_delta + epsilon) / (self.tot_delta_weight + epsilon))
        rms_gradient = np.sqrt((self.tot_sq_gradient + epsilon) / (self.tot_gradient_weight + epsilon))
        s = rms_delta / rms_gradient
        # Computes the delta.
        delta = s * g
        # Updates the delta average.
        self.tot_sq_delta = self.tot_sq_delta * decay + np.sum(np.square(delta))
        self.tot_delta_weight = self.tot_delta_weight * decay + 1.0
        # Finally, updates the weights.
        self.WLSTM -= delta * learning_factor
        return dX, g, dc0, dh0

    def backward_adadelta_vector_no_update(self, d, learning_factor=1.0, epsilon=0.0001, decay=0.95):
        """Implements backpropagation with the ADADELTA method, see
        http://arxiv.org/abs/1212.5701
        learning_factor indicates how much we should learn from this particular example."""
        dd = self._adapt_input_derivative(d)
        dX, g, dc0, dh0 = self._backward(dd)
        # Updates the gradient average.
        self.tot_sq_gradient = self.tot_sq_gradient * decay + np.sum(np.square(g))
        self.tot_gradient_weight = self.tot_gradient_weight * decay + 1.0
        # Computes the speed.
        rms_delta = np.sqrt((self.tot_sq_delta + epsilon) / (self.tot_delta_weight + epsilon))
        rms_gradient = np.sqrt((self.tot_sq_gradient + epsilon) / (self.tot_gradient_weight + epsilon))
        s = rms_delta / rms_gradient
        # Computes the delta.
        delta = s * g
        # Updates the delta average.
        self.tot_sq_delta = self.tot_sq_delta * decay + np.sum(np.square(delta))
        self.tot_delta_weight = self.tot_delta_weight * decay + 1.0
        # Finally, updates the weights.
        #self.WLSTM -= delta * learning_factor
        return dX, -delta * learning_factor


# -------------------
# TEST CASES
# -------------------

class BasicTests(unittest.TestCase):

    @unittest.skip("later")
    def test_checkSequentialMatchesBatch(self):
        """ check LSTM I/O forward/backward interactions """

        n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
        input_size = 10
        WLSTM = LSTM()
        WLSTM.initialize(input_size, d)  # input size, hidden size
        X = np.random.randn(n, b, input_size)
        h0 = np.random.randn(b, d)
        c0 = np.random.randn(b, d)

        # sequential forward
        cprev = c0
        hprev = h0
        caches = [{} for t in xrange(n)]
        Hcat = np.zeros((n, b, d))
        for t in xrange(n):
            xt = X[t:t + 1]
            _, cprev, hprev, cache = WLSTM._forward(xt, cprev, hprev)
            caches[t] = cache
            Hcat[t] = hprev

        # sanity check: perform batch forward to check that we get the same thing
        H, _, _, batch_cache = WLSTM._forward(X, c0, h0)
        assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

        # eval loss
        wrand = np.random.randn(*Hcat.shape)
        loss = np.sum(Hcat * wrand)
        dH = wrand

        # get the batched version gradients
        BdX, BdWLSTM, Bdc0, Bdh0 = WLSTM._backward(dH, batch_cache)

        # now perform sequential backward
        dX = np.zeros_like(X)
        dWLSTM = np.zeros_like(WLSTM.WLSTM)
        dc0 = np.zeros_like(c0)
        dh0 = np.zeros_like(h0)
        dcnext = None
        dhnext = None
        for t in reversed(xrange(n)):
            dht = dH[t].reshape(1, b, d)
            dx, dWLSTMt, dcprev, dhprev = WLSTM._backward(dht, caches[t], dcnext, dhnext)
            dhnext = dhprev
            dcnext = dcprev

            dWLSTM += dWLSTMt  # accumulate LSTM gradient
            dX[t] = dx[0]
            if t == 0:
                dc0 = dcprev
                dh0 = dhprev

        # and make sure the gradients match
        # print 'Making sure batched version agrees with sequential version: (should all be True)'
        self.assertTrue(np.allclose(BdX, dX))
        self.assertTrue(np.allclose(BdWLSTM, dWLSTM))
        self.assertTrue(np.allclose(Bdc0, dc0))
        self.assertTrue(np.allclose(Bdh0, dh0))


    @unittest.skip("later")
    def test_checkBatchGradient(self):
        """ check that the batch gradient is correct """

        # lets gradient check this beast
        n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
        input_size = 10
        WLSTM = LSTM()
        WLSTM.initialize(input_size, d)  # input size, hidden size
        X = np.random.randn(n, b, input_size)
        h0 = np.random.randn(b, d)
        c0 = np.random.randn(b, d)

        # batch forward backward
        H, Ct, Ht, cache = WLSTM._forward(X, c0, h0)
        wrand = np.random.randn(*H.shape)
        loss = np.sum(H * wrand)  # weighted sum is a nice hash to use I think
        dH = wrand
        dX, dWLSTM, dc0, dh0 = WLSTM._backward(dH, cache)

        def fwd():
            h, _, _, _ = WLSTM._forward(X, c0, h0)
            return np.sum(h * wrand)

        # now gradient check all
        delta = 1e-5
        rel_error_thr_warning = 1e-2
        rel_error_thr_error = 1
        tocheck = [X, c0, h0]
        grads_analytic = [dX, dc0, dh0]
        names = ['X', 'c0', 'h0']
        for j in xrange(len(tocheck)):
            mat = tocheck[j]
            dmat = grads_analytic[j]
            name = names[j]
            # gradcheck
            for i in xrange(mat.size):
                old_val = mat.flat[i]
                mat.flat[i] = old_val + delta
                loss0 = fwd()
                mat.flat[i] = old_val - delta
                loss1 = fwd()
                mat.flat[i] = old_val

                grad_analytic = dmat.flat[i]
                grad_numerical = (loss0 - loss1) / (2 * delta)

                if grad_numerical == 0 and grad_analytic == 0:
                    rel_error = 0  # both are zero, OK.
                    status = 'OK'
                elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                    rel_error = 0  # not enough precision to check this
                    status = 'VAL SMALL WARNING'
                else:
                    rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                    status = 'OK'
                    if rel_error > rel_error_thr_warning: status = 'WARNING'
                    if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'
                self.assertEqual(status, 'OK')

                # print stats
                # print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
                #       % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


class TestLearning(unittest.TestCase):

    unittest.skip("later")
    def make_sequence3d(self, length, p):
        """Makes a random sequence of 0 and 1 as a 3d input"""
        return (np.random.random((length,1,1))<p) & 1

    unittest.skip("later")
    def make_sequence2d(self, length, p):
        """Makes a random sequence of 0 and 1 as a 2d input"""
        return (np.random.random((length,1))<p) & 1

    unittest.skip("later")
    def test_forward_only_3d(self):
        M = 4
        K = 6
        net = LSTM()
        net.initialize(1, M)
        X = self.make_sequence3d(K, 0.0)
        print "X:", X.flatten()
        Y = net.forward(X)
        self.assertEqual(Y.ndim, 2)
        self.assertEqual(Y.shape, (1, 4))

    unittest.skip("later")
    def test_forward_only_2d(self):
        M = 4
        K = 6
        net = LSTM()
        net.initialize(1, M)
        X = self.make_sequence2d(K, 0.0)
        Y = net.forward(X)
        self.assertEqual(Y.ndim, 1)
        print "Y non-batch:", Y

    unittest.skip("later")
    def test_adapt_input_1d(self):
        net = LSTM()
        net.cache = {'n': 10, 'b': 1}
        net.hidden_size = 4
        dd = net._adapt_input_derivative(np.array([1, 2, 3, 4]))
        self.assertEqual(dd.shape, (10, 1, 4))
        self.assertEqual(np.sum(dd[:9, :, :]), 0)

    unittest.skip("later")
    def test_adapt_input_2d(self):
        net = LSTM()
        net.cache = {'n': 10, 'b': 5}
        net.hidden_size = 4
        dd = net._adapt_input_derivative(np.array([[1, 2, 3, 4],
                                                   [5, 6, 7, 8],
                                                   [2, 4, 6, 8],
                                                   [3, 4, 3, 2],
                                                   [5, 5, 4, 3]]))
        self.assertEqual(dd.shape, (10, 5, 4))
        self.assertEqual(np.sum(dd[:9, :, :]), 0)

    def _run_total_test(self, K, num_ones, M, N, use_adadelta=False):
        """Runs a test in which sequences of K elements must have
        one of the specified number of ones.  Do a total of N
        learning iterations.  The net has M memory cells."""
        net = LSTM()
        net.initialize(1, M)
        num_report = N / 10
        num_err = 0
        tot_err = 0.0
        tot_tgt = 0.0
        for i in range(N):
            X = self.make_sequence2d(K, (num_ones[0] * 1.0) / K)
            # print "X:", X.flatten()
            Y = net.forward(X)
            # print "Y", Y
            # Target and loss.
            yt = 1.0 if np.sum(X) in num_ones else 0.0
            # print "X, yt:", X, yt
            e = (Y[0] - yt) ** 2
            # print "e:", e, "yt:", yt, "y:", Y[0]
            d = np.zeros(Y.shape)
            d[0] = 2.0 * (Y[0] - yt)
            #print "Loss der:", d
            if use_adadelta:
                net.backward_adadelta(d)
            else:
                net.backward_momentum(d)
            num_err += 1
            tot_err += e
            tot_tgt += yt
            if num_err >= num_report:
                print "After %d iterations, avg tgt = %f, avg err = %f" % (i + 1, tot_tgt / num_err, tot_err / num_err)
                num_err = 0
                tot_err = 0.0
                tot_tgt = 0.0
        if num_err > 0:
            print "After %d iterations, avg tgt = %f, avg err = %f" % (i + 1, tot_tgt / num_err, tot_err / num_err)

    def test_has_one(self):
        print "At least one 1:"
        self._run_total_test(6, [1, 2, 3, 4, 5], 4, 1000)

    def test_has_only_one(self):
        print "Exactly one 1:"
        self._run_total_test(6, [1], 4, 10000)

    def test_has_one_or_three(self):
        print "Either one, or three 1's:"
        self._run_total_test(6, [1, 3], 6, 10000)




if __name__ == "__main__":
    unittest.main()
