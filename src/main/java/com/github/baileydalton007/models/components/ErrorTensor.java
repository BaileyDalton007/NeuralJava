package com.github.baileydalton007.models.components;

/**
 * Class to store a neural network's weight error tensor and biasError array
 * together.
 * 
 * @author Bailey Dalton
 */
public class ErrorTensor {
    // Stores the weight error tensor.
    private final double[][][] weightErrors;

    // Stores the bias error array.
    private final double[] biasErrors;

    /**
     * Constructor for ErrorTensors.
     * 
     * @param weightErrors The model's weight error 3D tensor to be stored.
     * @param biasErrors   The model's bias error array to be stored.
     */
    public ErrorTensor(double[][][] weightErrors, double[] biasErrors) {
        this.weightErrors = weightErrors;
        this.biasErrors = biasErrors;
    }

    /**
     * Getter for the weight error 3D tensor stored.
     * 
     * @return The model's weight error 3D tensor stored in the error tensor
     */
    public double[][][] getWeightErrors() {
        return weightErrors;
    }

    /**
     * Getter for the bias error array stored.
     * 
     * @return The model's bias array stored in the error tensor
     */
    public double[] getBiasErrors() {
        return biasErrors;
    }
}