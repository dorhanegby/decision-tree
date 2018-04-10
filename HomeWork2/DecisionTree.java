package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Capabilities;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;

}

public class DecisionTree implements Classifier {
	private Node rootNode;

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

	}
    
    @Override
	public double classifyInstance(Instance instance) {
		return 0.0;
    }


    // TODO: Ben
	/**
	 * Calculate Gini Index
	 * @param p - A set of probabilities
	 * @return The gini index of p
	 */
    private double calcGini(double[] p) {
		return 0.0;
	}

	// TODO: Dor
	/**
	 * Calculate Entropy
	 * @param p - A set of probabilities
	 * @return The Entropy of p
	 */
	private double calcEntropy(double[] p) {
		return 0.0;
	}

	// Not test, should be fine
	private boolean getActual(Instance instance) {
		return isRecurrence(instance.stringValue(instance.numAttributes() - 1));
	}

	private boolean isRecurrence(String actual) {
		return actual == "recurrence-events";
	}
    
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}



}
