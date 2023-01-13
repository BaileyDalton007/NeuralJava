package com.github.baileydalton007.utils;

import com.github.baileydalton007.exceptions.TensorOperationException;

/**
 * Utility class for tensor operations. Specifically vectors (arrays) and
 * matrices (2D arrays).
 * 
 * @author Bailey Dalton
 */
public class TensorOperations {
    /**
     * Adds two arrays of the same length together and returns an array of the added
     * values.
     * 
     * @param vec1 The first array to add
     * @param vec2 The second array to add
     * @return An array of the added values from the first and second arrays.
     */
    public static double[] addElements(double[] vec1, double[] vec2) {
        // Ensures that the arrays are the same length.
        if (vec1.length != vec2.length)
            throw new TensorOperationException("Arrays must be the same length to add together. They have the lengths "
                    + vec1.length + " and " + vec2.length + ".");

        double[] output = new double[vec1.length];

        // Iterate through each element in the two arrays to add them, and store them in
        // the output.
        for (int i = 0; i < output.length; i++) {
            output[i] = vec1[i] + vec2[i];
        }

        return output;
    }

    /**
     * Adds two matrices of the same length together and returns an matrix of the
     * added values.
     * 
     * @param m1 The first matrix to add
     * @param m2 The second matrix to add
     * @return An matrix of the added values from the first and second matrices.
     */
    public static double[][] addElements(double[][] m1, double[][] m2) {
        // Ensures that the arrays are the same length.
        if (m1.length != m2.length)
            throw new TensorOperationException(
                    "Matrices must be the same length to add together. They have the lengths "
                            + m1.length + " and " + m2.length + ".");

        double[][] output = new double[m1.length][];

        // Iterate through each row in the matrices.
        for (int i = 0; i < m1.length; i++) {
            // Ensures that the rows are the same length.
            if (m1[i].length != m2[i].length)
                throw new TensorOperationException(
                        "Matrices must be the same length to add together. Their index " + i + " rows have the lengths "
                                + m1[i].length + " and " + m2[i].length + ".");

            // Initalizes a new row in the output the same size as the two matrices's.
            output[i] = new double[m1[i].length];

            // Iterates through the elements in this row and adds them together.
            for (int j = 0; j < m1[i].length; j++)
                output[i][j] = m1[i][j] + m2[i][j];
        }

        return output;
    }

    /**
     * Subtracts one matrix's elements from another and returns a matrix of the new
     * values
     * 
     * @param m1 The matrix to be subtracted from
     * @param m2 The matrix to subtract
     * @return An matrix of the subtracted values from the first minus second
     *         matrix.
     */
    public static double[][] subtractElements(double[][] m1, double[][] m2) {
        // Ensures that the arrays are the same length.
        if (m1.length != m2.length)
            throw new TensorOperationException(
                    "Matrices must be the same length to add together. They have the lengths "
                            + m1.length + " and " + m2.length + ".");

        double[][] output = new double[m1.length][];

        // Iterate through each row in the matrices.
        for (int i = 0; i < m1.length; i++) {

            // Ensures that the rows are the same length.
            if (m1[i].length != m2[i].length)
                throw new TensorOperationException(
                        "Matrices must be the same length to subtract. Their index " + i + " rows have the lengths "
                                + m1[i].length + " and " + m2[i].length + ".");

            // Initalizes a new row in the output matrix.
            output[i] = new double[m1[i].length];

            // Iterates through the elements in this row and subtracts them.
            for (int j = 0; j < m1[i].length; j++)
                output[i][j] = m1[i][j] - m2[i][j];
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

    /**
     * Divides every element in the matrix by the passed in value.
     * 
     * @param m1    The matrix that will have it elements divided
     * @param value The value that will divide the elements in the matrix
     * @return A matrix where each element is the element from the input matrix
     *         divided by the input value.
     */
    public static double[][] divideByValue(double[][] m1, double value) {
        double[][] output = new double[m1.length][];

        // Iterate through each row in the matrix.
        for (int i = 0; i < output.length; i++) {

            // Creates an output row the same size as the respective row in m1.
            output[i] = new double[m1[i].length];

            // Iterates through each element in the row to divide it by the value and store.
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = m1[i][j] / value;
            }
        }

        return output;
    }

    /**
     * Divides every element in the 3D tensor by the passed in value.
     * 
     * @param m1    The 3D tensor that will have it elements divided
     * @param value The value that will divide the elements in the tensor
     * @return A 3D tensor where each element is the element from the input matrix
     *         divided by the input value.
     */
    public static double[][][] divideByValue(double[][][] m1, double value) {
        double[][][] output = new double[m1.length][][];

        // Iterate through the first dimension of the tensor.
        for (int i = 0; i < m1.length; i++) {

            // Initializes matrix.
            output[i] = new double[m1[i].length][];

            // Iterate through the second dimension of the tensor.
            for (int j = 0; j < m1[i].length; j++) {

                // Initializes array to be populated in the tensor.
                // I don't even know what I am saying anymore.
                output[i][j] = new double[m1[i][j].length];

                // Iterate through the third dimension of the tensor, multiplying each element
                // by the input value.
                for (int k = 0; k < m1[i][j].length; k++) {
                    output[i][j][k] = m1[i][j][k] / value;
                }
            }
        }

        return output;
    }

    /**
     * Multiplies every element in the matrix by the passed in value.
     * 
     * @param m1    The matrix that will have it elements multiplied
     * @param value The value that will multiply the elements in the matrix
     * @return A matrix where each element is the element from the input matrix
     *         multiplied by the input value
     */
    public static double[][] multiplyByValue(double[][] m1, double value) {
        double[][] output = new double[m1.length][];

        // Iterate through each row in the matrix.
        for (int i = 0; i < output.length; i++) {

            // Creates an output row the same size as the respective row in m1.
            output[i] = new double[m1[i].length];

            // Iterates through each element in the row to multiply it by the value and
            // store.
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = m1[i][j] * value;
            }
        }

        return output;
    }

    /**
     * Takes as input a vector and a matrix and multplies every element in the Nth
     * row of the matrix by the Nth element in the vector.
     * 
     * @param vec    The vector that has the same amount of elements as the matrix
     *               has rows
     * @param matrix The matrix that will have its rows multiplied by the respective
     *               vector value.
     * @return A matrix where each of the input matrix's rows have been multiplied
     *         by the corresponding element in the input vector.
     */
    public static double[][] multiplyRowsByVector(double[] vec, double[][] matrix) {
        // Makes sure the matrix and vector are compatible sizes.
        if (vec.length != matrix.length)
            throw new TensorOperationException(
                    "Vector must have the same amount of elements as there are rows in the matrix. The vector has a length of "
                            + vec.length + " and the matrix has " + matrix.length + " rows.");

        double[][] output = new double[vec.length][];

        // Iterates through each row in the matrix.
        for (int i = 0; i < matrix.length; i++) {
            output[i] = new double[matrix[i].length];

            // Iterates through each element in this row of the matrix, multiplying each
            // element by the current row's corresponding vector value.
            for (int j = 0; j < matrix[i].length; j++) {
                output[i][j] = matrix[i][j] * vec[i];
            }
        }

        return output;

    }
}