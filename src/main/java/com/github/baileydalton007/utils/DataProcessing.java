package com.github.baileydalton007.utils;

import com.github.baileydalton007.exceptions.IncompatibleInputException;

public class DataProcessing {
    public static double[][] oneHotEncode(double[][] data, int min, int max) {
        double[][] output = new double[data.length][];

        for (int i = 0; i < data.length; i++) {
            if (data[i].length != 1) {
                throw new IncompatibleInputException(
                        "Each target value must be a vector of size 1 to be one hot encoded. Expected a size of 1 but got a size of "
                                + data[i].length + "on example index " + i + ".");
            }

            output[i] = new double[(max - min) + 1];

            for (int j = 0; j < output[i].length; j++) {
                if (data[i][0] == j + min) {
                    output[i][j] = 1.0;
                } else {
                    output[i][j] = 0.0;
                }
            }
        }

        return output;
    }
}
