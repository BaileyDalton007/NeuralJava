package com.github.baileydalton007.activationfunctions;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class TestActivationFunctions {

    /**
     * Rounds an input (x) to n decimal places.
     * 
     * @param x The input value to be rounded
     * @param n The amount of decimal places to round the input value to
     * @return x rounded to n decimal places
     */
    private double roundNPlaces(double x, int n) {
        double scale = Math.pow(10, n);
        return Math.round(x * scale) / scale;
    }

    /**
     * Unit test for the sigmoid activation function.
     * 
     * Outputs are checked to 3 decimal places.
     * 
     * Tests 0 and a few positive and negative values.
     */
    @Test
    public void testSigmoidFunction() {
        SigmoidFunction sigmoid = new SigmoidFunction();

        // Tests 0.
        assertEquals(0.5, roundNPlaces(sigmoid.apply(0.0), 3));

        // Tests some intermediate values.
        assertEquals(0.622, roundNPlaces(sigmoid.apply(0.5), 3));
        assertEquals(0.378, roundNPlaces(sigmoid.apply(-0.5), 3));
        assertEquals(0.731, roundNPlaces(sigmoid.apply(1.0), 3));
        assertEquals(0.269, roundNPlaces(sigmoid.apply(-1.0), 3));

        // Tests large inputs that are rounded to 0 or 1.
        assertEquals(1.000, roundNPlaces(sigmoid.apply(100.0), 3));
        assertEquals(0.000, roundNPlaces(sigmoid.apply(-100.0), 3));

    }

}
