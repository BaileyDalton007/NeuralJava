package com.github.baileydalton007.models.components;

/**
 * Class to store a neural network's neuronError matrix and biasError array
 * together.
 * 
 * @author Bailey Dalton
 */
public class ErrorTensor {
    // Stores the neuron error matrix.
    private final double[][] neuronErrors;

    // Stores the bias error array.
    private final double[] biasErrors;

    /**
     * Constructor for ErrorTensors.
     * 
     * @param neuronErrors The model's neuron error matrix to be stored
     * @param biasErrors   The model's bias error array to be stored.
     */
    public ErrorTensor(double[][] neuronErrors, double[] biasErrors) {
        this.neuronErrors = neuronErrors;
        this.biasErrors = biasErrors;
    }

    /**
     * Getter for the neuron error matrix stored.
     * 
     * @return The model's neuron error matrix stored in the error tensor
     */
    public double[][] getNeuronErrors() {
        return neuronErrors;
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