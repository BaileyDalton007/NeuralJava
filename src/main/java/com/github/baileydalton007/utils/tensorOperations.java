package com.github.baileydalton007.utils;

import com.github.baileydalton007.exceptions.TensorOperationException;

/**
 * Utility class for tensor operations. Specifically vectors (arrays) and
 * matrices (2D arrays).
 * 
 * @author Bailey Dalton
 */
public class tensorOperations {
    /**
     * Adds two arrays of the same length together and returns an array of the added
     * values.
     * 
     * @param m1 The first array to add
     * @param m2 The second array to add
     * @return An array of the added values from the first and second arrays.
     */
    public static double[] addElements(double[] m1, double[] m2) {
        // Ensures that the arrays are the same length.
        if (m1.length != m2.length)
            throw new TensorOperationException("Arrays must be the same length to add together. They have the lengths "
                    + m1.length + " and " + m2.length + ".");

        double[] output = new double[m1.length];

        // Iterate through each element in the two arrays to add them, and store them in
        // the output.
        for (int i = 0; i < output.length; i++) {
            output[i] = m1[i] + m2[i];
        }

        return output;
    }

    /**
     * Multiplies two arrays of the same length together and returns an array of the
     * products.
     * 
     * @param m1 The first array to multiply
     * @param m2 The second array to multiply
     * @return An array of the multiplied values from the first and second arrays.
     */
    public static double[] multiplyElements(double[] m1, double[] m2) {
        // Ensures that the arrays are the same length.
        if (m1.length != m2.length)
            throw new TensorOperationException(
                    "Arrays must be the same length to multiply together. They have the lengths "
                            + m1.length + " and " + m2.length + ".");

        double[] output = new double[m1.length];

        // Iterate through each element in the two arrays to multply them, and store
        // them in the output.
        for (int i = 0; i < output.length; i++) {
            output[i] = m1[i] * m2[i];
        }

        return output;
    }
}