package com.company;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static int[] mainIndex;

    public static int count = 0;

    public static double[][] normalizedPTable;

    public static double[][] cumulativeF;

    public static double[] cumulativeZ;

    public static boolean posNeg = false;

    public static boolean negPos = false;

    public static void main(String[] args) {
	    // Binary AdaBoosting
	    // Step : Read the input file
        String file1Contents =  readFile("Input/Input3.txt");

        String[] fileLines = file1Contents.split("\\n");

        String[] line1 = fileLines[0].split(" ");

        int t = Integer.parseInt(line1[0]); // Number of Iterations

        int n = Integer.parseInt(line1[1]); // Number of Examples

        double e = Double.parseDouble(line1[2]); // Epsilon - a small number

        double[] x = new double[n]; // A list of n real numbers assumed to be in increasing order

        String[] line2 = fileLines[1].split(" ");

        for(int i = 0; i < n; i++) {
            x[i] = Double.parseDouble(line2[i]);
        }

        int[] y = new int[n]; // A list of n numbers that are either +1 or -1

        String[] line3 = fileLines[2].split(" ");

        for(int i = 0; i < n; i++) {
            y[i] = Integer.parseInt(line3[i]);
        }

        double[] p = new double[n]; // A list of n non negative numbers that sum up to 1

        ArrayList<ArrayList<Double>> updatedP = new ArrayList<ArrayList<Double>>();

        String[] line4 = fileLines[3].split(" ");

        for(int i = 0; i < n; i++) {
            p[i] = Double.parseDouble(line4[i]);
        }

        mainIndex = new int[t]; // Index position that separated negative and positive for each iteration

        normalizedPTable = new double[t+1][n];

        cumulativeF = new double[t][n]; // To store the cumulative f(x)

        cumulativeZ = new double[t];

        // Step : Perform binary Ada boosting for t iterations
        for(int k = 0; k < t; k++) {
            System.out.println("---");
            System.out.println("Iteration " + k);
            if(k > 0) {
                // pass the normalized p value calculated in the previous iteration
                for(int i = 0; i < p.length; i++) {
                    p[i] = normalizedPTable[k][i];
                }
            }

            /*for(double i : p)
                System.out.print(i + " ");
            System.out.println();*/

            double h = createWeakClassifiers(x, y, p);

            double alpha = 0.5 * Math.log((1-h)/h);
            System.out.println("3. Weight of H (alpha) is " + alpha);

            double qCorrect = Math.exp(-alpha);

            double qWrong = Math.exp(alpha);

            //System.out.println("Q-Correct = " + qCorrect + " Q-Wrong = " + qWrong);

            //int[] correctness = new int[n];
            int[] correctness = createCorrectnessMatrix(n, y, k);

            //System.out.println("Correctness ");
            /*for(int i = 0; i < n; i++) {
                System.out.print(correctness[i] + " ");
            }
            System.out.println();*/

            double[] preNormalizedP = findPreNormalized(mainIndex[0], correctness, p, qCorrect, qWrong);

            double z = 0.0;

            for(double i : preNormalizedP)
                z = z + i;

            System.out.println("4. The probabilities normalization factor Z is " + z);

            double[] normalizedP = new double[n];

            System.out.println("5. The probabilities after normalization p is ");
            for(int i = 0; i < p.length; i++) {
                normalizedP[i] = preNormalizedP[i]/z;
                System.out.print(normalizedP[i] + " ");
            }
            System.out.println();

            for(int i = 0; i < p.length; i++) {
                normalizedPTable[k+1][i] = normalizedP[i];
            }

            // Step : Build classifier matrix
            int[] classifier = createClassifierMatrix(n, k);

            // Step : Find cumulative f(x)
            double[] fx =  findCumulativeFx(alpha, classifier, k);

            System.out.println("6. The boosted classifier F(x) is ");
            for(int i = 0; i < n; i++) {
                cumulativeF[k][i] = fx[i];
                System.out.print(cumulativeF[k][i] + " ");
            }
            System.out.println();

            int errorOfClassifier = findBoostedClassifierError(fx, y);
            System.out.println("7. The error of boosted classifier E(t) is " + errorOfClassifier);

            double cumulativeZ = findCumulativeZ(z, k);
            System.out.println("8. The bound on E(t) is " + cumulativeZ);
        }
    }

    private static double findCumulativeZ(double z, int k) {
        double result = 0.0;
        if(k == 0) {
            result = z;
            cumulativeZ[0] = result;
        }

        if(k > 0) {
            result = cumulativeZ[k-1] * z;
            cumulativeZ[k] = result;
        }

        return result;
    }

    public static int findBoostedClassifierError(double[] fx, int[] y) {
        int errorCount = 0;
        for(int i = 0; i < y.length; i++) {
            if(fx[i] < 0 && y[i] != -1) {
                errorCount = errorCount + 1;
            }

            if(fx[i] >= 0 && y[i] != 1) {
                errorCount = errorCount + 1;
            }
        }
        return errorCount;
    }

    public static int[] createClassifierMatrix(int n, int k) {
        int[] result = new int[n];
        int index = mainIndex[k];
        if(posNeg) {
            for(int i = 0; i < n; i++) {
                if(i < index)
                    result[i] = 1;
                else
                    result[i] = -1;
            }
        } else {
            for(int i = 0; i < n; i++) {
                if(i < index)
                    result[i] = -1;
                else
                    result[i] = 1;
            }
        }
        return result;
    }

    public static int[] createCorrectnessMatrix(int n, int[] y, int k) {
        int[] result = new int[n];
        // Step : Update correctness matrix based on the selected hypothesis
        int index = mainIndex[k];

        if(negPos) {
            for(int i = 0; i < index; i++) {
                if(y[i] != -1) {
                    result[i] = -1;
                } else {
                    result[i] = 1;
                }
            }

            for(int i = index; i < n; i++){
                if(y[i] != 1) {
                    result[i] = -1;
                } else {
                    result[i] = 1;
                }
            }
        }

        if(posNeg){
            for(int i = 0; i < index; i++) {
                if(y[i] != 1) {
                    result[i] = -1;
                } else {
                    result[i] = 1;
                }
            }

            for(int i = index; i < n; i++){
                if(y[i] != -1) {
                    result[i] = -1;
                } else {
                    result[i] = 1;
                }
            }
        }
        return result;
    }

    public static double[] findCumulativeFx(double alpha, int[] classifier, int k) {
        double[] result = new double[classifier.length];
        for(int i = 0; i < classifier.length; i++) {
            result[i] = classifier[i] * alpha;
        }

        if(k > 0) {
            for(int i = 0; i < classifier.length; i++) {
                result[i] = result[i] + cumulativeF[k-1][i];
            }
            return result;
        } else {
            return result;
        }
    }

    public static double[] findPreNormalized(int index, int[] correctness, double[] p, double qCorrect, double qWrong) {
        int neg = 0;
        int pos = 0;
        double[] preNorm = new double[p.length];

        for(int i = 0; i < p.length; i++) {
            if(correctness[i] == -1) {
                preNorm[i] = p[i] * qWrong;
            } else {
                preNorm[i] = p[i] * qCorrect;
            }
        }

        //System.out.println("Pre Normalized p");
        /*for(int i = 0; i < p.length; i++)
            System.out.print(preNorm[i] + " ");
        System.out.println();*/

        return preNorm;
    }

    public static double createWeakClassifiers(double[] x, int[] y, double[] p) {
        double[] h1 = new double[p.length+1]; // stores error value for each hypothesis of weak classifier - / +
        double[] h2 = new double[p.length+1]; // stores error value for each hypothesis of weak classifier + / -
        double minH = Double.MAX_VALUE;
        int index = 0;
        for(int i = 0; i <= p.length; i++) {

            // Step : Classify left as negative
            int neg1 = 0;
            for(neg1 = 0; neg1 < i; neg1++) {
                if(y[neg1] != -1) {
                    h1[i] = h1[i] + p[neg1];
                }
            }

            // Step : Classify right as positive
            int pos1 = 0;
            for(pos1 = neg1; pos1 < p.length; pos1++) {
                if(y[pos1] != 1) {
                    h1[i] = h1[i] + p[pos1];
                }
            }

            // Step : Classify left as positive
            int pos2 = 0;
            for(pos2 = 0; pos2 < i; pos2++) {
                if(y[pos2] != 1) {
                    h2[i] = h2[i] + p[pos2];
                }
            }

            // Step : Classify right as negative
            int neg2 = 0;
            for(neg2 = pos2; neg2 < p.length; neg2++) {
                if(y[neg2] != -1) {
                    h2[i] = h2[i] + p[neg2];
                }
            }

            //System.out.println("H value " + h1[i]);
            //System.out.println("H value " + h2[i]);
            // keep comparing h with previous h values to find minimum h
            if(h1[i] < h2[i]) {
                if(h1[i] < minH) {
                    minH = h1[i];
                    index = i;
                    negPos = true;
                    posNeg = false;
                }
            } else {
                if(h2[i] < minH) {
                    minH = h2[i];
                    index = i;
                    posNeg = true;
                    negPos = false;
                }
            }
        }

        mainIndex[count] = index;
        count = count + 1;

        if(posNeg) {
            System.out.println("1. The selected Weak classifier is H(x) = I(x<" + (index-0.5)+")");
        }

        if(negPos) {
            System.out.println("1. The selected Weak classifier is H(x) = I(x>" + (index-0.5)+")");
        }

        System.out.println("Index is " + index);

        System.out.println("2. The error of H" + (index+1)+ " is " + minH);

        return minH;
    }

    // Reads the input file and return its contents as a String
    public static String readFile(String filename) {
        String contents = "";
        int count = 0;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filename));
            String line = null;
            try {
                while((line = br.readLine())!= null) {
                    if(count == 0) {
                        contents = contents + line;
                        count = count + 1;
                    } else {
                        contents = contents + "\n" + line;
                        count = count + 1;
                    }
                    System.out.println(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return contents;
    }
}
