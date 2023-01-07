package com.github.baileydalton007.models.components;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import com.github.baileydalton007.exceptions.MatrixTooSmallException;

/**
 * JUnit class for testing the weight matrix class.
 * 
 * @author Bailey Dalton
 */
public class TestWeightMatrix {

    @Test
    public void testWeightMatrix() {
        // Creates a weight matrix instance to test.
        WeightMatrix w1 = new WeightMatrix(2, 3);

        // Tests getMatrix method when the matrix is initialized.
        assertArrayEquals(new double[][] { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }, w1.getMatrix());

        // Tests getWeight method when the matrix is initialized.
        assertEquals(0.0, w1.getWeight(0, 1));
        assertEquals(0.0, w1.getWeight(1, 2));

        // Makes sure an exception is thrown if a dimension is initialized to 0.
        try {
            new WeightMatrix(0, 0);
            fail();
        } catch (MatrixTooSmallException e) {
            // Expected ouput, nothing happens.
        } catch (Exception e) {
            fail();
        }
    }
}
