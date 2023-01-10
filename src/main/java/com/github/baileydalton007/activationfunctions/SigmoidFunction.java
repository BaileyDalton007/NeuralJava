package com.github.baileydalton007.activationfunctions;

/**
 * Sigmoid activation function for deep learning neurons.
 * 
 * @author Bailey Dalton
 */
public class SigmoidFunction extends ActivationFunction {

    /**
     * Method to apply the sigmoid function to an input.
     * 
     * @param x Input to the sigmoid function
     * @return The output of the sigmoid function with x as an input.
     */
    @Override
    public double[] apply(double[] x) {
        double[] output = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            output[i] = 1 / (1 + Math.exp(-x[i]));
        }

        return output;
    }

    /**
     * Method to apply the derivative of the sigmoid function to an input array.
     * Used for back propagation error calculations.
     * 
     * @param x Input array to the sigmoid derivative function
     * @return The output array of the derivative sigmoid function that maps to the
     *         input array.
     */
    @Override
    public double[] derivative(double[] x) {
        // Create an array to store the outputs.
        double[] output = new double[x.length];

        // Store the y (outputs) of the normal sigmoid function.
        // y = sigmoid(x)
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
     * @param x Input to the sigmoid derivative function
     * @return The output of the derivative sigmoid function with x as an input.
     */
    @Override
    public double derivative(double x) {
        // Packs the input into a double array to pass to apply, then unpacks it.
        double y = apply(new double[] { x })[0];

        return y * (1 - y);
    }

    /**
     * Returns "sigmoid", the name of the activation function.
     * 
     * @return String representation of the activation function
     */
    @Override
    public String toString() {
        return "sigmoid";
    }

}
