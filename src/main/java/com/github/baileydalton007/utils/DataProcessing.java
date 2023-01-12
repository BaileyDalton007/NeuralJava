package com.github.baileydalton007.utils;

import com.github.baileydalton007.exceptions.IncompatibleInputException;

/**
 * Class for simple data processing tasks.
 * 
 * @author Bailey Dalton
 */
public class DataProcessing {

    /**
     * Performs one hot encoding on an array of scalar inputs.
     * 
     * @param data The array of inputs to encode
     * @param min  The minimum classification in the input
     * @param max  The maximum classification in the input
     * @return A matrix of one hot encoded values, each example will be an array
     *         with the size [(max - min) + 1]
     */
    public static double[][] oneHotEncode(double[][] data, int min, int max) {
        // Creates a matrix to store the output encoded data.
        double[][] output = new double[data.length][];

        // Iterate through each row in the matrix (each training example).
        for (int i = 0; i < data.length; i++) {

            // Throw an array if the input is not a scalar.
            if (data[i].length != 1) {
                throw new IncompatibleInputException(
                        "Each target value must be a vector of size 1 to be one hot encoded. Expected a size of 1 but got a size of "
                                + data[i].length + "on example index " + i + ".");
            }

            // Creates an index for each class betwenn min and max.
            output[i] = new double[(max - min) + 1];

            // Iterates through each index of the encoded output looking for a match.
            for (int j = 0; j < output[i].length; j++) {
                // If the value matches the current input, then set the value to 1.
                if (data[i][0] == j + min)
                    output[i][j] = 1.0;
            }
        }

        return output;
    }
}
