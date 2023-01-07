package com.github.baileydalton007.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.models.components.Layer;

/**
 * JUnit test for testing the dense neural network model.
 * 
 * @author Bailey Dalton
 */
public class TestDenseNeuralNetwork {

    /**
     * Test for the forward propagation algorithm on dense neural networks.
     */
    @Test
    public void testForwardPropagation() {
        // Creates a neural network to test.
        DenseNeuralNetwork m = new DenseNeuralNetwork(new Layer[] {
                new Layer(3, "relu"),
                new Layer(2, "relu"),
        });

        // Checks that are all outputs are zero when initialized, because all weights
        // are zero.
        assertArrayEquals(new double[] { 0.0, 0.0 }, m.ForwardPropagation(new double[] { 1.0, 2.0, 3.0 }));

        // Tests that an exception is thrown if input array and the input layer are not
        // the same size.
        try {
            m.ForwardPropagation(new double[] { 1.0, 2.0 });
            fail();
        } catch (IncompatibleInputException e) {
            // Expected output, nothing happens.
        } catch (Exception e) {
            fail();
        }
    }
}
