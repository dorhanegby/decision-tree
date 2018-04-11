package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
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
		selectionMethod = SelectionMethod.GINI;
		buildTree(data);
    }
    public int majorityClass (Instances data) throws Exception
	{
		double [] p = getProbabilties(data);
		return (p[0] > p[1]) ? NO_RECURRENCE : RECURRENCE;
    }

    private Node buildTree(Instances data) throws Exception {
		Node node = new Node();
		// TODO: Need to make a stopping condition if all the data is of one class (recurrence, no_recurrence) @ Ben
		if(data.numAttributes() == 0) {
			// TODO: Mark node returnValue as the majority class
			return node;
		}

		// TODO: implemented Find best attribute @ Dor
		Attribute splittingAttribute = findSplittingCriterion(data);
		node.attributeIndex = splittingAttribute.index();

		Instances[] splitGroups = splitByCriterion(data, splittingAttribute);

		Instances newData = removeAttribute(data, splittingAttribute);

		return node;

	}

	private Instances[] splitByCriterion (Instances data, Attribute criterion) throws Exception {
		Instances [] instances = new Instances[criterion.numValues()];
		for(int i=0;i<instances.length;i++) {
			instances[i] = filterByAttributeValue(data, criterion, new int[] { i + 1 });
		}

		return instances;
	}
	// Not tested yet
	private Instances removeAttribute(Instances data, Attribute attribute) throws Exception {
		Remove remove = new Remove();

		remove.setAttributeIndices("" + attribute.index());
		remove.setInvertSelection(false);
		remove.setInputFormat(data);
		Instances newData = Filter.useFilter(data, remove);

		return newData;
	}

	private Attribute findSplittingCriterion(Instances data) throws Exception {
		int maxIndex = 0;
		double maxGain = 0;
		for (int i=0;i<data.numAttributes() - 1;i++) {
			double gain = calcGain(data, i);
			System.out.println("Gain: " + data.attribute(i).name() + "= " + gain);
			if(maxGain < gain) {
				maxIndex = i;
				maxGain = gain;
			}
		}

		return data.attribute(maxIndex);
	}
    
    @Override
	public double classifyInstance(Instance instance) {
		return 0.0;
    }


    private double calcGain(Instances data, int attributeIndex) throws Exception {
		return calcMeasure(data) - calcMeasureAttribute(data, data.attribute(attributeIndex));
	}

	/**
	 * Calculate Gini Index
	 * @param p - A set of probabilities
	 * @return The gini index of p
	 */
    private double calcGini(double[] p) {
		double sum = 0.0;
		for (int i = 0; i < p.length; i++) {
			sum += p[i] * p[i];
		}
		return 1 - sum;
	}

	/**
	 * Calculate Entropy
	 * @param p - A set of probabilities
	 * @return The Entropy of p
	 */
	private double calcEntropy(double[] p) {
		double sum = 0.0;
		for (int i = 0; i < p.length; i++) {
			if (p[i] != 0) {
				sum += p[i] * Math.log(p[i]);
			}
		}

		return sum;
	}

	private double calcMeasure(Instances data) throws Exception {
		if (selectionMethod == SelectionMethod.ENTROPY) {
			return calcEntropy(getProbabilties(data));
		} else {
			return calcGini(getProbabilties(data));
		}
	}

	private double calcMeasureAttribute(Instances data, Attribute attribute) throws Exception {
		double sum = 0.0;

		int attributeDiscreteValues = attribute.numValues();

		for(int i=0;i<attributeDiscreteValues;i++) {
			Instances filteredData = filterByAttributeValue(data, attribute, new int[] {(i + 1)});
			if(filteredData.size() != 0) {
                double weight = (filteredData.size() / (double) data.size());
                if(selectionMethod == SelectionMethod.ENTROPY) {
                	sum += weight * calcEntropy(getProbabilties(filteredData));
				}
				else
				{
					sum += weight * calcGini(getProbabilties(filteredData));
				}
            }
		}

		return sum;
	}

	private Instances getNoRecurrenceClass(Instances data) throws Exception {
		return filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( NO_RECURRENCE + 1 )});
	}

	private Instances getRecurrenceClass(Instances data) throws Exception {
		return filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( RECURRENCE + 1 )});
	}

	private double[] getProbabilties(Instances data) throws Exception{
		double[] probabilities = new double[2];
		probabilities[NO_RECURRENCE] = getNoRecurrenceClass(data).size() / (double) data.size();
		probabilities[RECURRENCE] = getRecurrenceClass(data).size() / (double) data.size();
        return probabilities;
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
