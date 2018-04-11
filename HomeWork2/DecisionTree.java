package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.SubsetByExpression;

import java.util.concurrent.Callable;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex; // splitting criterion
	double returnValue; // majority

}

public class DecisionTree implements Classifier {
    private final int NO_RECURRENCE = 0;
    private final int RECURRENCE = 1;
	private Node rootNode;
	private SelectionMethod selectionMethod;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		rootNode = new Node();
		selectionMethod = SelectionMethod.ENTROPY;
    }
    
    @Override
	public double classifyInstance(Instance instance) {
		return 0.0;
    }


    private double calcGain(Instances data, int attributeIndex) {
		return 0;
	}


    // TODO: Ben
	/**
	 * Calculate Gini Index
	 * @param p - A set of probabilities
	 * @return The gini index of p
	 */
    private double calcGini(double[] p) {
		double sum = 0.0;
		for(int i = 0 ; i < p.length;i++){
			sum += p[i]*p[i];
		}
		return sum;
	}

	/**
	 * Calculate Entropy
	 * @param p - A set of probabilities
	 * @return The Entropy of p
	 */
	private double calcEntropy(double[] p) {
		double sum = 0.0;
		for(int i=0;i<p.length;i++) {
		    if(p[i] != 0) {
                sum += p[i] * Math.log(p[i]);
            }
		}

		return 1- sum;
	}

	private double calcMeasureAttribute(Instances data, Attribute attribute) throws Exception {
		double sum = 0.0;

		int attributeDiscreteValues = attribute.numValues();

		for(int i=0;i<attributeDiscreteValues;i++) {
			Instances filteredData = filterByAttributeValue(data, attribute, new int[] {(i + 1)});
			if(filteredData.size() != 0) {
                sum += (filteredData.size() / (double) data.size());
                if(selectionMethod == SelectionMethod.ENTROPY) {
                	sum *= calcEntropy(getProbabilties(filteredData));
				}
				else
				{
					sum *= calcGini(getProbabilties(filteredData));
				}
            }
		}

		return sum;
	}

	private double[] getProbabilties(Instances data) throws Exception{
		double[] probabilities = new double[2];
		probabilities[NO_RECURRENCE] = filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( NO_RECURRENCE + 1 )}).size() / (double) data.size();
		probabilities[RECURRENCE] = filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( RECURRENCE + 1 )}).size() / (double) data.size();
        return probabilities;
	}

	// Not test, should be fine
	private boolean getActual(Instance instance) {
		return isRecurrence(instance.stringValue(instance.numAttributes() - 1));
	}

	private String flatArrayValues(int [] array) {
		String string = "";
		for(int i=0;i<array.length - 1;i++) {
			string += array[i] + ",";
		}

		string += array[array.length - 1];

		return string;
	}

	private Instances filterByAttributeValue(Instances dataToFilter, Attribute attribute, int[] valueIndecies) throws Exception {
        RemoveWithValues filter = new RemoveWithValues();
        String[] options = new String[5];
        options[0] = "-C";   // attribute index
		options[1] = "" + (attribute.index() + 1);
		options[2] = "-L" ;
		options[3] = flatArrayValues(valueIndecies);
		options[4] = "-V";
		filter.setOptions(options);

		filter.setInputFormat(dataToFilter);
		Instances newData = Filter.useFilter(dataToFilter, filter);
		return newData;
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

	private enum SelectionMethod {
		GINI,
		ENTROPY
	}




}
