# Graph-LSTM

This repository contains several pieces of code that are useful for applying machine learning to graphs. 

## DNN
**dnn.py** provides an implementation of deep neural networks.  The input consists in fixed-length feature vectors.

## LSTM
**lstm.py** provides an implementation of LSTMs.  The input consists in sequences of fixed-length feature vectors.

## Multi-Level LSTM
**multi_level_lstm.py** provides an implementation of multi-level LSTMs (see https://sites.google.com/view/ml-on-structures for papers and information).  The input consists in trees of nodes; each node has a feature vector.  The trees can be obtained, among other ways, by unrolling the local neighborhood of a node in a graph.

