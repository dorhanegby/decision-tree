package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("HomeWork2/Data/cancer_train.txt");
		Instances testingCancer = loadData("HomeWork2/Data/cancer_test.txt");
		Instances validationCancer = loadData("HomeWork2/Data/cancer_validation.txt");

		DecisionTree decisionTree = new DecisionTree();
		decisionTree.buildClassifier(trainingCancer);

        //TODO: complete the Main method
	}
}
