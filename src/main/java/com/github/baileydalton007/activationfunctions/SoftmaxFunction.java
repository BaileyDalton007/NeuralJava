package com.github.baileydalton007.activationfunctions;

/**
 * Softmax activation function for deep learning neurons.
 * Used for multiple class classification output layers.
 * 
 * @author Bailey Dalton
 */
public class SoftmaxFunction extends ActivationFunction {
    /**
     * Method to apply the softmax function to an array of inputs.
     * 
     * @param x Inputs to the softmax function
     * @return The output array of the softmax function mapped to the input array.
     */
    @Override
    public double[] apply(double[] x) {
        // Create an array that will be output.
        double[] output = new double[x.length];

        // Calculates the sum of e^all inputs to be used as the denominator of the
        // softmax function.
        double sum = 0.0;
        for (double z : x) {
            sum += Math.exp(z);
        }

        // Interates through each input and calculates the softmax probability.
        for (int i = 0; i < x.length; i++) {
            output[i] = Math.exp(x[i]) / sum;
        }

        return output;
    }

    /**
     * Method to apply the derivative of the softmax function to an input array.
     * Used for back propagation error calculations.
     * 
     * @param x Input array to the softmax derivative function
     * @return The output array of the softmax function that maps to the input
     *         array.
     */
    @Override
    public double[] derivative(double[] x) {
        // Create an array to store the outputs.
        double[] output = new double[x.length];

        // Store the y (outputs) of the normal sigmoid function.
        // y = softmax(x)
        double[] y = apply(x);

        // For each y, calculate the derivative and store it in the output.
        for (int i = 0; i < x.length; i++) {
            output[i] = y[i] * (1 - y[i]);
        }

        return output;
    }

    /**
     * Method to apply the derivative of the sigmoid function to an input.
     * Used for back propagation error calculations.
     * 
     * SHOULD ONLY BE USED FOR BIAS ERROR CALCULATIONS.
     * 
     * @param x Input to the softmax derivative function
     * @return The output of the derivative softmax function with x as an input.
     */
    @Override
    public double derivative(double x) {
        // Packs the input into a double array to pass to apply, then unpacks it.
        double y = apply(new double[] { x })[0];

        return y * (1 - y);
    }

    /**
     * Returns "Softmax", the name of the activation function.
     * 
     * @return String representation of the activation function
     */
    @Override
    public String toString() {
        return "softmax";
    }
}
