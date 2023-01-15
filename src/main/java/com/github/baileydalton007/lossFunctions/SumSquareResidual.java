package com.github.baileydalton007.lossFunctions;

import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.activationfunctions.ActivationFunction;

/**
 * Loss function class that describles the sum squared residual loss function.
 * 
 * @author Bailey Dalton
 */
public class SumSquareResidual extends lossFunction {

    /**
     * Finds the loss of a current training example using sum squared residual.
     * 
     * @param currLayer          The current layer that loss is being calculated on
     * @param nextLayerErrorSums The summation of the errors in the next layer
     * @param target             The target value for the current input
     * @param outputLayer        If true this layer will be treated as an output
     *                           layer, else it will be treated as a hidden layer
     * @return An array storing the loss of each neuron in the current layer
     */
    public static double[] loss(Layer currLayer, double[] nextLayerErrorSums, double[] target, boolean outputLayer) {
        double[] output = new double[currLayer.size()];

        // Stores the activations for the current layer.
        double[] activations = currLayer.getLayerActivations();

        // Calculates the derivative terms for the error calculations below. Represented
        // as f'(WEIGHTED_SUMS)
        double[] derivativeTerms = currLayer.getActivationFunction().derivative(currLayer.getLayerInputs());

        // For each neuron in the current layer, calculate the loss and store it in the
        // output.
        for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {
            if (outputLayer) {
                // LOSS = f'(WEIGHTED_SUMS) * (ACTIVAION - TARGET)
                // This is just for output layers.
                output[neuronIndex] = derivativeTerms[neuronIndex]
                        * (activations[neuronIndex] - target[neuronIndex]);

            } else {
                // LOSS = f'(WEIGHTED_SUMS) * SUM(ERRORS_IN_NEXT_LAYER)
                // This is for all hidden layers.
                output[neuronIndex] = derivativeTerms[neuronIndex]
                        * nextLayerErrorSums[neuronIndex];
            }
        }
        return output;
    }

    public static double lossBias(double[] neuronLoss, double biasValue, double[] target, Layer currLayer,
            boolean outputLayer) {

        double errorSum = 0.0;

        if (outputLayer) {
            double[] activations = currLayer.getLayerActivations();

            // Iterates thought the output neurons, summing the differences between the
            // activations and the target values.
            for (int i = 0; i < currLayer.size(); i++) {
                errorSum += activations[i] - target[i];
            }

        } else {

            // Sums all the errors in the neurons in the current layer.
            for (int i = 0; i < neuronLoss.length; i++) {
                errorSum += neuronLoss[i];
            }

        }

        return currLayer.getActivationFunction().derivative(biasValue) * errorSum;

    }
}
