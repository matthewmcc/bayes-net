import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Simple class for approximate inference based on the Poker-game network.
 */
public class BayesNet {

	/**
	 * Inner class for representing a node in the network.
	 */
	private class Node {

		// The name of the node
		private String name;

		// The parent nodes
		private Node[] parents;

		// The probabilities for the CPT
		private double[] probs;

		// The current value of the node
		public boolean value;

		// The children nodes
		private List<Node> children = new ArrayList<Node>();
		
		/**
		 * Initializes the node.
		 */
		private Node(String n, Node[] pa, double[] pr) {
			name = n;
			parents = pa;
			probs = pr;
		}

		/**
		 * Returns conditional probability of value "true" for the current node
		 * based on the values of the parent nodes.
		 * 
		 * @return The conditional probability of this node, given its parents.
		 */
		private double conditionalProbability() {

			int index = 0;
			for (int i = 0; i < parents.length; i++) {
				if (parents[i].value == false) {
					index += Math.pow(2, parents.length - i - 1);
				}
			}
			return probs[index];
		}
		/**
		 * Sets the children array of the node
		 */
		private void setChildren() {

			for(Node n : nodes) {
				for(Node nC : nodes) {
					for(Node nP : nC.parents) {
						if(nP == n && !n.children.contains(nC))
							n.children.add(nC);
					}
				}
			}
		}
	}

	// The list of nodes in the Bayes net
	private Node[] nodes;

	// A collection of examples describing whether Bot B is { cocky, bluffing }
	public static final boolean[][] BBLUFF_EXAMPLES = { { true, true },
			{ true, true }, { true, true }, { true, false }, { true, true },
			{ false, false }, { false, false }, { false, true },
			{ false, false }, { false, false }, { false, false },
			{ false, false }, { false, true } };

	/**
	 * Constructor that sets up the Poker-game network.
	 */
	public BayesNet() {

		nodes = new Node[7];

		nodes[0] = new Node("B.Cocky", new Node[] {}, new double[] { 0.05 });
		nodes[1] = new Node("B.Bluff", new Node[] { nodes[0] },
				calculateBBluffProbabilities(BBLUFF_EXAMPLES));
		nodes[2] = new Node("A.Deals", new Node[] {},
				new double[] { 0.5 });
		nodes[3] = new Node("A.GoodHand", new Node[] { nodes[2] },
				new double[] { 0.75, 0.5 });
		nodes[4] = new Node("B.GoodHand", new Node[] { nodes[2] },
				new double[] { 0.4, 0.5 });
		nodes[5] = new Node("B.Bets", new Node[] { nodes[1], nodes[4] },
				new double[] { 0.95, 0.7, 0.9, 0.01 });
		nodes[6] = new Node("A.Wins", new Node[] { nodes[3], nodes[4] },
				new double[] { 0.45, 0.75, 0.25, 0.55 });
	}

	/**
	 * Prints the current state of the network to standard out.
	 */
	public void printState() {

		for (int i = 0; i < nodes.length; i++) {
			if (i > 0) {
				System.out.print(", ");
			}
			System.out.print(nodes[i].name + " = " + nodes[i].value);
		}
		System.out.println();
	}

	/**
	 * Calculates the probability that Bot B will bluff based on whether it is
	 * cocky or not.
	 * 
	 * @param bluffInstances
	 *            A set of training examples in the form { cocky, bluff } from
	 *            which to compute the probabilities.
	 * @return The probability that Bot B will bluff when it is { cocky, !cocky
	 *         }.
	 */
	public double[] calculateBBluffProbabilities(boolean[][] bluffInstances) {
		
		double[] probabilities = new double[2];
		double cocky = 0, cockyBluff = 0, notCocky = 0, notCockyBluff = 0;

		for(boolean[] b : bluffInstances) {
			if(b[0]) {
				cocky++;
				if(b[1]) cockyBluff++;
			}
			else {
				notCocky++;
				if (b[1]) notCockyBluff++;
			}
		}

		probabilities[0] = cockyBluff / cocky;
		probabilities[1] = notCockyBluff / notCocky;
		
		return probabilities;
	}

	/**
	 * This method calculates the exact probability of a given event occurring,
	 * where all variables are assigned a given evidence value.
	 *
	 * @param evidenceValues
	 *            The values of all nodes.
	 * @return -1 if the evidence does not cover every node in the network.
	 *         Otherwise a probability between 0 and 1.
	 */
	public double calculateExactEventProbability(boolean[] evidenceValues) {
		// Only performs exact calculation for all evidence known.
		if (evidenceValues.length != nodes.length)
			return -1;
	
		// Sets evidence values
		for(int i = 0; i < nodes.length; i++)
			nodes[i].value = evidenceValues[i];
		
		double probs[] = new double[nodes.length];
		int index = 0;

		for (Node n : nodes) {
			if(n.value)
				probs[index] = n.conditionalProbability();
			else probs[index] = 1 - n.conditionalProbability();
			index++;
		}
		
		// Calculates final probability, checks if it's inbounds and returns it.
		double finalProb = 1;
		for(double p : probs) 
			finalProb *= p;
		
		if(finalProb >= 0 && finalProb <= 1)
			return finalProb;
		return -1;
	}
	
	/**
	 * Normalises the vector given and returns the result
	 */
	public double normalise(double[] vector) {
		return vector[0] / (vector[0] + vector[1]);
	}

	/**
	 * This method assigns new values to the nodes in the network by sampling
	 * from the joint distribution 	(based on PRIOR-SAMPLE method from text
	 * book/slides).
	 */
	public void priorSample() {
		Random r = new Random();
		for(Node n : nodes)
			n.value = r.nextDouble() < n.conditionalProbability();
	}

	/**
	 * Rejection sampling. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see REJECTION-SAMPLING method from text
	 * book/slides).
	 * 
	 * The nodes/variables are specified by their indices in the nodes array.
	 * The array evidenceValues has one value for each index in
	 * indicesOfEvidenceNodes. See also examples in main().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double rejectionSampling(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {
		
		double[] supportVector = {0, 0};
		
		for(int j = 0; j < N; j++) {
			priorSample();
			boolean holds = true;
			for(int i = 0; i < evidenceValues.length && holds; i++) {
				if (nodes[indicesOfEvidenceNodes[i]].value != evidenceValues[i])
					holds = false;
			}
			
			// Sums vector values given queryNode value
			if(holds) {
				if(nodes[queryNode].value)
					supportVector[0] += 1;
				else supportVector[1] += 1;
			}
		}

		return normalise(supportVector);
	}

	/**
	 * This method assigns new values to the non-evidence nodes in the network
	 * and computes a weight based on the evidence nodes (based on
	 * WEIGHTED-SAMPLE method from text book/slides).
	 * 
	 * The evidence is specified as in the case of rejectionSampling().
	 * 
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @return The weight of the event occurring.
	 * 
	 */
	public double weightedSample(int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues) {

		double weight = 1;
		int index = 0;
		Random r = new Random();	
		
		for(int i = 0; i < nodes.length; i++) {
			index = Arrays.binarySearch(indicesOfEvidenceNodes, i);
			if(index >= 0) {
				nodes[i].value = evidenceValues[index];
				
				// Computes product of weights vector for nodes
				if(nodes[i].value)
					weight *= nodes[i].conditionalProbability();
				else weight *= (1 - nodes[i].conditionalProbability());
			}
			else
				nodes[i].value = r.nextDouble() < nodes[i].conditionalProbability();
		}
		return weight;
	}

	/**
	 * Likelihood weighting. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see LIKELIHOOD-WEIGHTING method from text
	 * book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double likelihoodWeighting(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {
		
		double weightVector[] = {0, 0};
		double weight = 0;
		
		for(int i = 0; i < N; i++) {
			weight = weightedSample(indicesOfEvidenceNodes, evidenceValues);
			
			// Sums vector weights given queryNode value
			if(nodes[queryNode].value)
				weightVector[0] += weight;
			else weightVector[1] += weight;
		}
		
		return normalise(weightVector);
	}
	
	/**
	 * MCMC inference. Returns probability of query variable being true given
	 * the values of the evidence variables, estimated based on the given total
	 * number of samples (see MCMC-ASK method from text book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double MCMCask(int queryNode, int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues, int N) {
		
		double mVector[] = {0, 0};
		Random r = new Random();
		
		// Create list of non evidence nodes
		int nonEvidenceNodes[] = new int[nodes.length - indicesOfEvidenceNodes.length];
		int index  = 0;
		for(int i = 0; i < nodes.length; i++) {
			if(Arrays.binarySearch(indicesOfEvidenceNodes, i) < 0) {
				nonEvidenceNodes[index] = i;
				index++;
			}
		}
		
		// Set evidence values
		for(int i = 0; i < indicesOfEvidenceNodes.length; i++)
			nodes[indicesOfEvidenceNodes[i]].value = evidenceValues[i];
		
		// Sets the values of children in nodes
		for(Node n : nodes)
			n.setChildren();
		
		// Initializes nodes in nonEvidenceNodes with random value
		for(int nE : nonEvidenceNodes)
			nodes[nE].value = r.nextDouble() < nodes[nE].conditionalProbability();
		
		for(int i = 0; i < N; i++) {
			// Sums vector values given queryNode value
			if(nodes[queryNode].value)
				mVector[0]++;
			else mVector[1]++;
			
			// Samples all values in nonEvidenceNodes given Markov Blanket
			for(int nE : nonEvidenceNodes) {
				markovBlacketSample(nE, r);
			}
		}

		return normalise(mVector);
	}
	
	// Given the nodes index this function samples the...
	// ...nodes value given the markov blanket of the node in the BN
	public void markovBlacketSample(int nE, Random r) {
		double p, pC;
		double pVector[] = {0, 0};
		
		// Calculates the true value of the MB equation 
		nodes[nE].value = true;
		p = nodes[nE].conditionalProbability();
		pC = 1;
		
		// Computes the product of childrens CP
		for(Node nC : nodes[nE].children) {
			if(nC.value) pC *= nC.conditionalProbability();
			else pC *= (1 -  nC.conditionalProbability());
		}
		pVector[0] = p * pC;
		
		// Calculates the false value of the equation 
		nodes[nE].value = false;
		p = 1 - nodes[nE].conditionalProbability();
		pC = 1;
		
		// Computes the product of childrens CP
		for(Node nC : nodes[nE].children) {
			if(nC.value) pC *= nC.conditionalProbability();
			else pC *= (1 -  nC.conditionalProbability());
		}
		pVector[1] = p * pC;
		
		p = normalise(pVector);
		
		// Samples new value
		nodes[nE].value = r.nextDouble() < p;
	}
	
	/**
	 * Similar to MCMCask. But only one node in the nonEvidenceNodes is sampled
	 * on every iteration. This makes the algorithm a lot more efficient to run
	 * on smaller bayesian networks.
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double GibbsAsk(int queryNode, int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues, int N) {
		
		double mVector[] = {0, 0};
		
		// Create list of non evidence nodes
		int nonEvidenceNodes[] = new int[nodes.length - indicesOfEvidenceNodes.length];
		int index  = 0;
		for(int i = 0; i < nodes.length; i++) {
			if(Arrays.binarySearch(indicesOfEvidenceNodes, i) < 0) {
				nonEvidenceNodes[index] = i;
				index++;
			}
		}
		
		// Set evidence values
		for(int i = 0; i < indicesOfEvidenceNodes.length; i++)
			nodes[indicesOfEvidenceNodes[i]].value = evidenceValues[i];
		
		// Sets the values of children in nodes
		for(Node n : nodes)
			n.setChildren();
		
		// Initializes nodes in nonEvidenceNodes with random value
		Random r = new Random();
		for(int nE : nonEvidenceNodes)
			nodes[nE].value = r.nextDouble() < nodes[nE].conditionalProbability();
		
		for(int i = 0; i < N; i++) {
			if(nodes[queryNode].value)
				mVector[0]++;
			else mVector[1]++;
			
			// Samples all values in nonEvidenceNodes given Markov Blanket
			markovBlacketSample(nonEvidenceNodes[r.nextInt(nonEvidenceNodes.length)], r);
		}
		
		return normalise(mVector);
	}

	/**
	 * The main method, with some example method calls.
	 */
	public static void main(String[] ops) {

		// Create network.
		BayesNet b = new BayesNet();

		double[] bluffProbabilities = b
				.calculateBBluffProbabilities(BBLUFF_EXAMPLES);
		System.out.println("When Bot B is cocky, it bluffs "
				+ (bluffProbabilities[0] * 100) + "% of the time.");
		System.out.println("When Bot B is not cocky, it bluffs "
				+ (bluffProbabilities[1] * 100) + "% of the time.");

		double bluffWinProb = b.calculateExactEventProbability(new boolean[] {
				true, true, true, false, false, true, false });
		System.out
				.println("The probability of Bot B winning on a cocky bluff "
						+ "(with bet) and both bots have bad hands (A dealt) is: "
						+ bluffWinProb);

		// Sample five states from joint distribution and print them
		for (int i = 0; i < 5; i++) {
			b.priorSample();
			b.printState();
		}

		// Print out results of some example queries based on rejection
		// sampling, likelyhood waiting, MCMCask and GibbsAsk.
		System.out.println("");
		System.out.println("Rejection Sampling");
		// Probability of B.GoodHand given bet and A not win.
		System.out.println(b.rejectionSampling(4, new int[] { 5, 6 },
				new boolean[] { true, false }, 1000000));

		// Probability of betting given a cocky
		System.out.println(b.rejectionSampling(1, new int[] { 0 },
				new boolean[] { true }, 1000000));

		// Probability of B.Goodhand given B.Bluff and A.Deal
		System.out.println(b.rejectionSampling(4, new int[] { 1, 2 },
				new boolean[] { true, true }, 1000000));
		
		System.out.println("");
		System.out.println("Likelyhood Weighting");
		// Probability of B.GoodHand given bet and A not win.
		System.out.println(b.likelihoodWeighting(4, new int[] { 5, 6 },
				new boolean[] { true, false }, 1000000));

		// Probability of betting given a cocky
		System.out.println(b.likelihoodWeighting(1, new int[] { 0 },
				new boolean[] { true }, 1000000));

		// Probability of B.Goodhand given B.Bluff and A.Deal
		System.out.println(b.likelihoodWeighting(4, new int[] { 1, 2 },
				new boolean[] { true, true }, 1000000));
		
		System.out.println("");
		System.out.println("MCMCask");
		// Probability of B.GoodHand given bet and A not win.
		System.out.println(b.MCMCask(4, new int[] { 5, 6 },
				new boolean[] { true, false }, 1000000));

		// Probability of betting given a cocky
		System.out.println(b.MCMCask(1, new int[] { 0 },
				new boolean[] { true }, 100000));

		// Probability of B.Goodhand given B.Bluff and A.Deal
		System.out.println(b.MCMCask(4, new int[] { 1, 2 },
				new boolean[] { true, true }, 100000));
		
		System.out.println("");
		System.out.println("GibbsAsk");
		// Probability of B.GoodHand given bet and A not win.
		System.out.println(b.GibbsAsk(4, new int[] { 5, 6 },
				new boolean[] { true, false }, 10000000));

		// Probability of betting given a cocky
		System.out.println(b.GibbsAsk(1, new int[] { 0 },
				new boolean[] { true }, 1000000));

		// Probability of B.Goodhand given B.Bluff and A.Deal
		System.out.println(b.GibbsAsk(4, new int[] { 1, 2 },
				new boolean[] { true, true }, 1000000));
	}
}