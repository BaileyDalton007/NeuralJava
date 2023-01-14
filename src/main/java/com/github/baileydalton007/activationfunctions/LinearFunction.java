package com.github.baileydalton007.activationfunctions;

/**
 * Linear activation function for deep learning neurons.
 * 
 * @author Bailey Dalton
 */
public class LinearFunction extends ActivationFunction {

    /**
     * Method to apply the linear function to a layer of inputs.
     * 
     * @param x Input array to the linear function
     * @return The output array of the linear function with outputs mapping to the
     *         input array.
     */
    @Override
    public double[] apply(double[] x) {

        // The joys of OOP :)
        return x;
    }

    /**
     * Method to apply the derivative of the linear function to an input array.
     * Used for back propagation error calculations.
     * 
     * 
     * @param x Input array to the linear derivative function
     * @return The output array of the derivative linear function that maps to the
     *         input array.
     */
    @Override
    public double[] derivative(double[] x) {
        // Create an array to store the outputs.
        double[] output = new double[x.length];

        // For each input, "calculate" the derivative and store it in the output.
        for (int i = 0; i < x.length; i++) {
            output[i] = 1;
        }

        return output;

    }

    /**
     * Method to apply the derivative of the linear function to an input.
     * Used for back propagation error calculations.
     * 
     * @param x Input to the linear derivative function
     * @return The output of the derivative linear function with x as an input.
     */
    @Override
    public double derivative(double x) {
        return 1.0;
    }

    /**
     * Returns "linear", the name of the activation function.
     * 
     * @return String representation of the activation function
     */
    @Override
    public String toString() {
        return "linear";
    }
}