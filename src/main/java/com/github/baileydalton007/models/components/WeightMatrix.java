package com.github.baileydalton007.models.components;

import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.MatrixTooSmallException;

/**
 * Class for a matrix specifically for representing the weights between a layer
 * and the previous layer.
 * 
 * The index system for this matrix may seem a bit backwards, but it is
 * derived from the mathematical way the weight matrix is represented.
 * 
 * A weight matrix is used to represent the weights of connections between the
 * neurons in layer L and the neurons in layer (L-1). The resulting shape of the
 * matrix is (Amount of Neurons in Layer L, Amount of Neurons in Layer (L-1)).
 * 
 * The weight connecting the neuron in layer L at index i to the neuron in layer
 * (L-1) at index j is represented in the matrix as WeightMatrix[i][j].
 * 
 * @author Bailey Dalton
 */
public class WeightMatrix {
    // 2D array for storing the weight matrix.
    private double[][] matrix;

    /**
     * The constructor to initialize a weight matrix.
     * Matrix with be of shape (numNeurons, prevNumNeurons)
     * 
     * @param numNeurons     The number of neurons in the layer, must be > 0
     * @param prevNumNeurons The number of neurons in the previous layer, must be >
     *                       0
     */
    public WeightMatrix(int numNeurons, int prevNumNeurons) {
        // If either dimenstion of the matrix is 0, throw an exception.
        if (numNeurons < 1 || prevNumNeurons < 1)
            throw new MatrixTooSmallException();

        matrix = new double[numNeurons][prevNumNeurons];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = 0.1;
            }
        }
    }

    /**
     * Getter for the entire weight matrix.
     * 
     * @return The weight matrix as an 2D array
     */
    public double[][] getMatrix() {
        return matrix;
    }

    /**
     * Getter for a weight from the weight matrix using the index of the neuron in
     * the current layer and the index of the connected neuron in the previous
     * layer.
     * 
     * @param neuronIndex     The index of the neuron in this layer.
     * @param prevNeuronIndex The index of the neuron in the previous layer.
     * @return The weight of the connection between the neuron in the previous layer
     *         and the neuron in this layer.
     */
    public double getWeight(int neuronIndex, int prevNeuronIndex) {
        return getMatrix()[neuronIndex][prevNeuronIndex];
    }

    /**
     * Setter for the whole matrix.
     * 
     * @param input The 2D array to make the matrix
     */
    public void setMatrix(double[][] input) {
        matrix = input;
    }

    /**
     * Subtracts the input matrix from this matrix.
     * Matrices must be the same shape to subtract.
     * 
     * @param input WeightMatrix of the same shape as this matrix to subtract
     */
    public void subtract(WeightMatrix input) {
        // Stores the matrices to subtract.
        double[][] m1 = this.getMatrix();
        double[][] m2 = input.getMatrix();

        // Checks if the two matrices are the same shape.
        if (m1.length != m2.length || m1[0].length != m2[0].length)
            throw new IncompatibleInputException("Weight Matrices must have the same dimensions to subtract");

        // Iterates through each row of the matrix.
        for (int i = 0; i < m1.length; i++) {

            // Iterates through each column of the matrix subtracting the elements.
            for (int j = 0; j < m1[0].length; j++) {
                m1[i][j] -= m2[i][j];
            }
        }

        this.setMatrix(m1);
    }
}
