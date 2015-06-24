priorSample() generates network states that are distributed according to the joint distribution specified by the network.

Use priorSample() to implement the rejection sampling method in rejectionSampling().

weightedSample() samples values for the non-evidence variables and returns a weight based on the values of the evidence variables.

Use weightedSample() to implement likelihood weighting in the method called likelihoodWeighting().

Implement Markov chain Monte Carlo inference in the MCMCask() method. To this end you may want to add a method that computes the conditional probability of a node's value based on the node's Markov blanket (which is given on one of the slides provided).

Gibbs ASK is a the final method that uses Markov Blanket and Monte Carlo inference to compute network conditional probability.