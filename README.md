# Graph-LSTM

This repository contains several pieces of code that are useful for applying machine learning to graphs. 
See [project page](https://sites.google.com/view/ml-on-structures) for the overall project, papers, and data. 

Many prediction problems can be phrased as inferences over local neighborhoods of graphs.  The graph represents the interaction between entities, and the neighborhood of each entity contains information that allows the inferences or predictions. 
This project enables the application of machine learning directly to such graph neighborhoods, allowing predictions to be learned from examples, bypassing the step of creating and tuning an inference model or summarizing the neighborhoods via a fixed set of hand-crafted features.
The approach is based on a multi-level architecture built from Long Short-Term Memory neural nets (LSTMs); the LSTMs learn how to summarize the neighborhood from data.

## How it works

The code performs predictions for one ``target'' graph node at a time. 
First, the graph is unfolded from the target node, yielding a tree with the target node as its root at level 0, its neighbors as level-1 children, its neighbors' neighbors as level-2 children, and so forth, up to a desired depth D.
At each tree node v of level 0 <= d < D, a level-d+1 LSTM is fed sequentially the information from the children of v at level d+1, and produces as output information for v itself. 
Thus, we exploit LSTMs' ability to process sequences of any length to process trees of any branching factor.
The top-level LSTM produces the desired prediction for the target node. 
The architecture requires training D LSTMs, one per tree level.
The LSTMs learn how to summarize the neighborhood up to radius $D$ on the basis of data, avoiding the manual task of synthesizing a fixed set of features.
By dedicating one LSTM to each level, we can tailor the learning (and the LSTM size) to the distance from the target node.

## Code included

This repository contains various ML algorithms, which can be used independently or in combination.

### DNN
**dnn.py** provides an implementation of deep neural networks.  The input consists in fixed-length feature vectors.

### LSTM
**lstm.py** provides an implementation of LSTMs.  The input consists in sequences of fixed-length feature vectors.

### Multi-Level LSTM
**multi_level_lstm.py** provides an implementation of multi-level LSTMs (see the [project page](https://sites.google.com/view/ml-on-structures) for papers and information).  The input consists in trees of nodes; each node has a feature vector.  The trees can be obtained, among other ways, by unrolling the local neighborhood of a node in a graph.

### MLSLNN
**mlslnn.py** is a helper function to apply multi-level LSTMs to a graph or tree.  The code defined in multi_level_lstm.py enables the summarization of the feature vectors of a tree rooted at v into an output vector (from the top-level LSTM) y. The vector y summarizes the features of the children of v (and subtrees rooted there), but not of v itself.  Thus, it is useful to combine the vector y, and the feature vector f(v) of v, via a top-level neural network that gives the overall output.  The class MLSLNN enables this. 

## Contributors

* [Luca de Alfaro](https://sites.google.com/a/ucsc.edu/luca/)
* Rakshit Agrawal
* Vassilis Polychonopoulos
