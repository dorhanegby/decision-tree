package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.SubsetByExpression;

import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.jar.Attributes;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex; // splitting criterion
	double returnValue; // majority
	int recurrence;
	int noRecurrence;
	Instances data;
	String attType;
}

public class DecisionTree implements Classifier {
    private final int NO_RECURRENCE = 0;
    private final int RECURRENCE = 1;
	private Node rootNode;
	private SelectionMethod selectionMethod;
	private HashMap<Attribute, Integer> attributeToIndex;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		selectionMethod = SelectionMethod.GINI;
		attributeToIndex = createAttributeMapping(data);
		this.rootNode = buildTree(data);
		int recurrence = 0;
		int no_recurrence = 0;
		for(int i=0;i<data.size();i++){
			if(classifyInstance(data.get(i)) == 1) {
				if(data.get(i).classValue() + 1 != 1) {
					System.out.println("mistake! " + i);
					System.out.println(data.get(i));
				}
				recurrence++;
			}
			else {
				if(data.get(i).classValue() + 1 != 2) {
					System.out.println("mistake! " + i);
					System.out.println(data.get(i));
				}
				no_recurrence++;
			}


		}

		System.out.println("recurrence: " + recurrence);
		System.out.println("no recurrence: " + no_recurrence);
	}


    private Node buildTree(Instances data) throws Exception {
		Node node = new Node();
		node.recurrence = getRecurrenceClass(data).size();
		node.noRecurrence = getNoRecurrenceClass(data).size();
		node.data = data;
		if(isTheSameClass(data)){
			node.returnValue = majorityClass(data);
			return node;
		}
		if(data.numAttributes() == 0) {
			node.returnValue = majorityClass(data);
			return node;
		}
		Attribute splittingAttribute = findSplittingCriterion(data);
		node.attributeIndex = attributeToIndex.get(splittingAttribute);
		node.returnValue = majorityClass(data);
		Instances[] splitGroups = splitByCriterion(data, splittingAttribute);
		node.children = new Node[splitGroups.length];
		for(int i=0;i<splitGroups.length;i++) {
			if(splitGroups[i].size() != 0) {
				node.children[i] = buildTree(splitGroups[i]);
				node.children[i].parent = node;
				node.children[i].attType = splittingAttribute.value(i);
			}
			else {
				Node childnode = new Node();
				node.children[i] = childnode;
				childnode.parent = node;
				childnode.returnValue = node.returnValue;
				childnode.attType = splittingAttribute.value(i);
			}
		}

		return node;

	}

	private HashMap<Attribute, Integer> createAttributeMapping(Instances data) {
		HashMap<Attribute, Integer> hashMap = new HashMap<>();
		for(int i =0;i<data.numAttributes();i++) {
			hashMap.put(data.attribute(i), i);
		}

		return hashMap;
	}

	private int majorityClass (Instances data) throws Exception
	{
		double [] p = getProbabilties(data);
		return (p[0] > p[1]) ? NO_RECURRENCE : RECURRENCE;
	}

	private Instances[] splitByCriterion (Instances data, Attribute criterion) throws Exception {
		Instances [] instances = new Instances[criterion.numValues()];
		for(int i=0;i<instances.length;i++) {
			instances[i] = filterByAttributeValue(data, criterion, new int[] { i + 1 });
			instances[i] = removeAttribute(instances[i], criterion);
		}

		return instances;
	}
	private boolean isTheSameClass(Instances data) throws Exception {
		double[] p = getProbabilties(data);
		boolean isHomogeneous = false;
		if (p[0] == 1 || p[1] == 1) {
			isHomogeneous = true;
		}
		return isHomogeneous;
	}

	private Instances removeAttribute(Instances data, Attribute attribute) throws Exception {
		Remove remove = new Remove();

		remove.setAttributeIndices("" + (attribute.index() + 1));
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
			if(maxGain < gain) {
				maxIndex = i;
				maxGain = gain;
			}
		}

		return data.attribute(maxIndex);
	}
    
    @Override
	public double classifyInstance(Instance instance) {
		Node cureNode = rootNode;
		while (cureNode.children != null) {
			cureNode = cureNode.children[instance.attribute(cureNode.attributeIndex).indexOfValue(instance.stringValue(cureNode.attributeIndex))];
		}
		return cureNode.returnValue;
	}


    private double calcGain(Instances data, int attributeIndex) throws Exception {
		return calcMeasure(data) - calcMeasureAttribute(data, data.attribute(attributeIndex));
	}

	/**
	 * Calculate Gini Index
	 * @param p - A set of probabilities
	 * @return The Gini index of p
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

		return -sum;
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
		int noRecurrenceClassIndex = 2;
		return filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( noRecurrenceClassIndex )});
	}

	private Instances getRecurrenceClass(Instances data) throws Exception {
		int recurrenceClassIndex = 1;
		return filterByAttributeValue(data, data.attribute(data.classIndex()), new int[] {( recurrenceClassIndex )});
	}

	private double[] getProbabilties(Instances data) throws Exception{
		double[] probabilities = new double[2];
		Instances test = getNoRecurrenceClass(data);
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
