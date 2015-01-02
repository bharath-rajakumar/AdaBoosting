//package com.company;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static int[] mainIndex;

    public static int[] mainIndexReal;

    public static int count = 0;

    public static double[][] normalizedPTable;

    public static double[][] cumulativeF;

    public static double[][] cumulativeRealF;

    public static double[] cumulativeZ;

    public static double[] cumulativeRealZ;

    public static boolean posNeg = false;

    public static boolean negPos = false;

    public static double[][] normalizedRealPTable;

    public static double[][] gTable;

    public static String binClassifier;

    public static void main(String[] args) {
        //Note : Comment binary ada boosting if you want to run Real ada boosting and vice versa

	    // Step : Read the input file
        String file1Contents =  readFile("Input/adaboost-1.dat");

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
        binClassifier = "";






        // Step : Perform binary Ada boosting for t iterations
        for(int k = 0; k < t; k++) {
            System.out.println();
            System.out.println("Iteration " + (k+1));
            if(k > 0) {
                // pass the normalized p value calculated in the previous iteration
                for(int i = 0; i < p.length; i++) {
                    p[i] = normalizedPTable[k][i];
                }
            }

            for(double i : p)
                System.out.print(i + " ");
            System.out.println();

            double h = createWeakClassifiers(x,y, p);

            double alpha = 0.5 * Math.log((1-h)/h);
            System.out.println("Alpha  = " + alpha);

            double qCorrect = Math.exp(-alpha);

            double qWrong = Math.exp(alpha);

            //System.out.println("Q-Correct = " + qCorrect + " Q-Wrong = " + qWrong);

            //int[] correctness = new int[n];
            int[] correctness = createCorrectnessMatrix(n, y, k);

            //System.out.println("Correctness ");
            for(int i = 0; i < n; i++) {
                System.out.print(correctness[i] + " ");
            }
            System.out.println();

            double[] preNormalizedP = findPreNormalized(mainIndex[0], correctness, p, qCorrect, qWrong);

            double z = 0.0;

            for(double i : preNormalizedP)
                z = z + i;

            System.out.println("Normalization Factor Z = " + z);

            double[] normalizedP = new double[n];

            System.out.print("Pi after normalization = ");
            for(int i = 0; i < p.length; i++) {
                normalizedP[i] = preNormalizedP[i]/z;
                if(i == p.length - 1) {
                    System.out.print(normalizedP[i]);
                } else {
                    System.out.print(normalizedP[i] + ", ");
                }
            }
            System.out.println();

            for(int i = 0; i < p.length; i++) {
                normalizedPTable[k+1][i] = normalizedP[i];
            }

            // Step : Build classifier matrix
            int[] classifier = createClassifierMatrix(n, k);

            // Step : Find cumulative f(x)
            double[] fx =  findCumulativeFx(alpha, classifier, k);
            for(int i = 0; i < fx.length; i++) {
                cumulativeF[k][i] = fx[i];
            }


            binClassifier = alpha + "*" + binClassifier + "+";
            System.out.println("Boosted Classifier f(x) = " + binClassifier.substring(0,binClassifier.length()-1));
            for(int i = 0; i < n; i++) {
                cumulativeF[k][i] = fx[i];
                if(i == n - 1) {
                    System.out.print(cumulativeF[k][i]);
                } else {
                    System.out.print(cumulativeF[k][i] + ", ");
                }
            }
            System.out.println();

            int errorOfClassifier = findBoostedClassifierError(fx, y);
            System.out.println("Boosted classifier Error = " + errorOfClassifier);

            double cumulativeZ = findCumulativeZ(z, k);
            System.out.println("Bound on Error = " + cumulativeZ);
        }





        // Step : Real Ada boosting
        posNeg = false;
        negPos = false;

        normalizedRealPTable = new double[t+1][n];

        mainIndexReal = new int[t]; // Index position that separated negative and positive for each iteration

        gTable = new double[t][2]; // Store all cTPlus and cTMinus for each iteration

        cumulativeRealF = new double[t][n]; // Stores all the cumulative Fx for each iteration and for each example

        cumulativeRealZ = new double[t];

        for(int k = 0; k < t; k++) {
            System.out.println();
            System.out.println("Iteration " + (k+1));

            if(k > 0) {
                // pass the normalized p value calculated in the previous iteration
                for(int i = 0; i < p.length; i++) {
                    p[i] = normalizedRealPTable[k][i];
                }
            }

            double hReal = createRealWeakClassifier(y, p, x, e, k);

            System.out.println("G error = " + hReal);

            System.out.println("C_Plus = " + gTable[k][0] + ", C_Minus = " + gTable[k][1]);
            // Step : Build the pre normalized table
            double[] preNormalized = findRealPreNormalized(gTable[k][0], gTable[k][1], y, p, k);

            // Step : Find normalization factor Z(t)
            double z = 0.0;

            for(double a : preNormalized)
                z = z + a;
            System.out.println("Normalization Factor Z = " + z);

            //Step : Build the normalized table
            double[] normalizedP = findRealNormalized(z, preNormalized);

            System.out.print("Pi after normalization = ");

            for(int i = 0; i < normalizedP.length; i++) {
                if(i == n - 1) {
                    System.out.print(normalizedP[i]);
                } else {
                    System.out.print(normalizedP[i] + ", ");
                }
            }

            System.out.println();

            for(int i = 0; i < p.length; i++) {
                normalizedRealPTable[k+1][i] = normalizedP[i];
            }

            // Step : Store the normalized p, which will be used for the next iteration
            for(int i = 0; i < p.length; i++) {
                normalizedRealPTable[k+1][i] = normalizedP[i];
            }

            // Step : Build the classfier matrix
            int[] classifier = createRealClassifierMatrix(n, k);

            // Step : find cumulative Fx
            double[] fx =  findRealCumulativeFx(gTable[k][0], gTable[k][1], classifier, k);

            System.out.print("f(x) = ");
            for(int i = 0; i < n; i++) {
                cumulativeRealF[k][i] = fx[i];
                if(i == n - 1) {
                    System.out.print(cumulativeRealF[k][i]);
                } else {
                    System.out.print(cumulativeRealF[k][i] + ", ");
                }
            }
            System.out.println();

            double error = findBoostedRealClassifierError(fx, y);
            System.out.println("Boosted Classifier Error = " + error);

            double cumulativeZ = findCumulativeZ(z, k);
            System.out.println("Bound on Error = " + cumulativeZ);
        }
        //Real Ada boosting ends here
    }

    private static double findBoostedRealClassifierError(double[] fx, int[] y) {
        int error = 0;

        for(int i = 0; i < y.length; i++) {
            if(fx[i] < 0 && y[i] != -1)
                error = error + 1;

            if(fx[i] >= 0 && y[i] != 1)
                error = error + 1;
        }

        double result = error/(double)y.length;

        return result;
    }

    private static double[] findRealCumulativeFx(double cTPlus, double cTMinus, int[] classifier, int k) {
        double[] result = new double[classifier.length];

        for(int i = 0; i < classifier.length; i++) {
            if(classifier[i] == -1) {
                result[i] = cTMinus;
            } else {
                result[i] = cTPlus;
            }
        }

        if(k > 0) {
            for(int i = 0; i < classifier.length; i++) {
                result[i] = result[i] + cumulativeRealF[k - 1][i];
            }
            return result;
        } else {
            return result;
        }

    }

    public static int[] createRealClassifierMatrix(int n, int k) {
        int[] result = new int[n];
        int index = mainIndexReal[k];
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

    private static double[] findRealNormalized(double z, double[] preNormalized) {
        double[] normalized = new double[preNormalized.length];
        for(int i = 0; i < preNormalized.length; i++) {
            normalized[i] = preNormalized[i]/z;
        }
        return normalized;
    }

    public static double[] findRealPreNormalized(double cTPlus, double cTMinus, int[] y, double[] p, int k) {
        double[] preNormalized = new double[y.length];

        // compute the prenormalized p based on the chosen hypothesis
        int[] h = new int[y.length];
        int index = mainIndexReal[k];

        if(posNeg) {
            for(int i = 0; i < index; i++) {
                h[i] = 1;
            }

            for(int i = index; i < y.length; i++) {
                h[i] = -1;
            }
        }

        if(negPos) {
            for(int i = 0; i < index; i++) {
                h[i] = -1;
            }

            for(int i = index; i < y.length; i++) {
                h[i] = 1;
            }
        }

        for(int i = 0; i < y.length; i++) {
            if(h[i] == 1) {
                preNormalized[i] = p[i] * Math.exp(-1 * y[i] * cTPlus);
            }

            if(h[i] == -1) {
                preNormalized[i] = p[i] * Math.exp(-1 * y[i] * cTMinus);
            }
        }

        return preNormalized;
    }

    public static double findCumulativeZ(double z, int k) {
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


        if(posNeg) {
            if(index == 0) {
                System.out.println("Classifier h = I(x < " + (x[index] - 0.5) +")");
                binClassifier = binClassifier + "I(x < " + (x[index] - 0.5) +")";
            } else if (index == x.length -1) {
                System.out.println("Classifier h = I(x < " + (x[index] + 0.5) +")");
                binClassifier = binClassifier + "I(x < " + (x[index] + 0.5) +")";
            } else {
                System.out.println("Classifier h = I(x < " + (x[index] + x[index - 1])/2 +")");
                binClassifier = binClassifier + "I(x < " + (x[index] + x[index - 1])/2 +")";
            }
        }

        if(negPos) {
            if(index == 0) {
                System.out.println("Classifier h = I(x > " + (x[index] - 0.5) +")");
                binClassifier = binClassifier + "I(x > " + (x[index] - 0.5) +")";
            } else if(index == x.length -1){
                System.out.println("Classifier h = I(x > " + (x[index] + 0.5) +")");
                binClassifier = binClassifier + "I(x > " + (x[index] + 0.5) +")";
            } else {
                System.out.println("Classifier h = I(x > " + (x[index] + x[index - 1])/2 +")");
                binClassifier = binClassifier + "I(x > " + (x[index] + x[index - 1])/2 +")";
            }
        }

        count = count + 1;
        //System.out.println("Index is " + index);

        System.out.println("Error = " + minH);

        return minH;
    }

    public static double createRealWeakClassifier(int[] y, double[] p, double[] x, double epsilon, int k) {
        double minG = Double.MAX_VALUE;
        int index = 0;
        double pRPlus = 0;
        double pWPlus = 0;
        double pRMinus = 0;
        double pWMinus = 0;

        for(int i = 0; i <= p.length; i++) {

            int[] hyp1 = new int[p.length]; // hypothesis matrix with -/+
            int[] hyp2 = new int[p.length]; // hypothesis matrix with +/-

            // build the first hypothesis matrix with -/+
            for(int j = 0; j < i; j++) {
                hyp1[j] = -1;
            }

            for(int j = i; j < p.length; j++) {
                hyp1[j] = 1;
            }

            // build the second hypothesis matrix with +/-
            for(int j = 0; j < i; j++) {
                hyp2[j] = 1;
            }

            for(int j = i; j < p.length; j++) {
                hyp2[j] = -1;
            }

            // compute G for hypothesis1
            double pRPlus1 = 0;
            double pRMinus1 = 0;
            double pWPlus1 = 0;
            double pWMinus1 = 0;

            for(int j = 0; j < p.length; j++) {
                if(y[j] == -1) {
                    if(y[j] == hyp1[j]) {
                        pRMinus1 = pRMinus1 + p[j];
                    } else {
                        pWMinus1 = pWMinus1 + p[j];
                    }
                }

                if(y[j] == 1) {
                    if(y[j] == hyp1[j]) {
                        pRPlus1 = pRPlus1 + p[j];
                    } else {
                        pWPlus1 = pWPlus1 + p[j];
                    }
                }
            }

            double g1 = Math.sqrt(pRPlus1*pWMinus1) + Math.sqrt(pRMinus1*pWPlus1);

            // compute G for hypothesis2
            double pRPlus2 = 0;
            double pRMinus2 = 0;
            double pWPlus2 = 0;
            double pWMinus2 = 0;

            for(int j = 0; j < p.length; j++) {
                if(y[j] == 1) {
                    if(y[j] == hyp2[j]) {
                        pRPlus2 = pRPlus2 + p[j];
                    } else {
                        pWPlus2 = pWPlus2 + p[j];
                    }
                }

                if(y[j] == -1) {
                    if(y[j] == hyp2[j]) {
                        pRMinus2 = pRMinus2 + p[j];
                    } else {
                        pWMinus2 = pWMinus2 + p[j];
                    }
                }
            }

            double g2 = Math.sqrt(pRPlus2*pWMinus2) + Math.sqrt(pRMinus2*pWPlus2);

            // find the minimum G value and compare it with previous iteration's minimum G
            if(g1 < g2) {
                if (g1 < minG) {
                    minG = g1;
                    negPos = true;
                    posNeg = false;
                    index = i;
                    pRPlus = pRPlus1;
                    pWPlus = pWPlus1;
                    pRMinus = pRMinus1;
                    pWMinus = pWMinus1;
                }
            } else {
                if(g2 < minG) {
                    minG = g2;
                    posNeg = true;
                    negPos = false;
                    index = i;
                    pRPlus = pRPlus2;
                    pWPlus = pWPlus2;
                    pRMinus = pRMinus2;
                    pWMinus = pWMinus2;
                }
            }
        }

        mainIndexReal[k] = index;
        //System.out.println("Index is " + index);

        double cTPlus = 0.5 * Math.log((pRPlus + epsilon)/(pWMinus + epsilon));
        double cTMinus = 0.5 * Math.log((pWPlus + epsilon)/(pRMinus + epsilon));

        if(posNeg) {
            System.out.println("Classifier h = I(x < " +(x[index] + x[index - 1])/2 +")");
        }

        if(negPos) {
            System.out.println("Classifier h = I (x > " +(x[index] + x[index - 1])/2 +")");
        }

        //store the result
        gTable[k][0] = cTPlus;
        gTable[k][1] = cTMinus;

        return minG;
    }

    private static double computeG(int[] y, double[] p, int[] hyp) {
        double g = 0;
        double pRPlus = 0;
        double pRMinus = 0;
        double pWPlus = 0;
        double pWMinus = 0;

        // compare hypothesis matrix with input matrix y to find the 4 probabilities error
        for(int j = 0; j < p.length; j++) {
            if(y[j] == 1) {
                if(y[j] == hyp[j]) {
                    pRPlus = pRPlus + p[j];
                } else {
                    pWPlus = pWPlus + p[j];
                }
            }

            if(y[j] == -1) {
                if(y[j] == hyp[j]) {
                    pRMinus = pRMinus + p[j];
                } else {
                    pWMinus = pWMinus + p[j];
                }
            }
        }

        g = Math.sqrt(pRPlus*pWMinus) + Math.sqrt(pRMinus*pWPlus);

        return g;
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
