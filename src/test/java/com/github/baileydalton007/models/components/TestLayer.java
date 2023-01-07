package com.github.baileydalton007.models.components;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.LayerTooSmallException;

/**
 * JUnit class for testing the layer class.
 * 
 * @author Bailey Dalton
 */
public class TestLayer {

    /**
     * Test for layer's size method.
     */
    @Test
    public void testSize() {
        assertEquals(1, new Layer(1, "relu").size());
        assertEquals(4, new Layer(4, "relu").size());
        assertEquals(16, new Layer(16, "relu").size());
        assertEquals(128, new Layer(128, "relu").size());

        // Makes sure an exception is thrown if layer is too small.
        try {
            new Layer(0, "relu").size();
        } catch (LayerTooSmallException e) {
            // Expected output, nothing happens.
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Test for layer's neuron getter.
     * Since all neurons are initialized the same and abstracted, the test just
     * makes sure a neuron is returned.
     */
    @Test
    public void testGetNeuron() {
        Layer layer = new Layer(16, "relu");

        assertTrue(layer.getNeuron(0) instanceof Neuron);
        assertTrue(layer.getNeuron(5) instanceof Neuron);
        assertTrue(layer.getNeuron(15) instanceof Neuron);

        // Makes sure an exception is thrown neuron is outside of layer.
        try {
            assertTrue(layer.getNeuron(16) instanceof Neuron);
        } catch (IndexOutOfBoundsException e) {
            // Expected output, nothing happens.
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Tests layer's input and getLayerActivations methods.
     */
    @Test
    public void testInputAndActivation() {
        // Creates an layer instance to test on.
        Layer l = new Layer(3, "relu");

        // Makes sure that all neurons have an activation of 0 before recieving input.
        assertArrayEquals(new double[] { 0.0, 0.0, 0.0 }, l.getLayerActivations());

        // Passes in some inputs to the array and makes sure their activations match.
        // They will be equal if positive with the ReLU activation function.
        l.input(new double[] { 1.0, 1.0, 1.0 });
        assertArrayEquals(new double[] { 1.0, 1.0, 1.0 }, l.getLayerActivations());

        // Ensures that ReLU activation function (max(0, x)) is applied to inputs.
        l.input(new double[] { -2.0, 3.0, -1.0 });
        assertArrayEquals(new double[] { 0.0, 3.0, 0.0 }, l.getLayerActivations());

        // Makes sure that an exception is thrown if input size does not match the size
        // of the layer.
        try {
            l.input(new double[] { -2.0, 3.0 });
        } catch (IncompatibleInputException e) {
            // Expected output, nothing happens.
        } catch (Exception e) {
            fail();
        }

        // Tests to make sure the sigmoid activation function works as well.
        Layer s = new Layer(2, "sigmoid");

        // Makes sure that all neurons have an activation of 0.5 before recieving input.
        // sigmoid(0) = 0.5
        assertArrayEquals(new double[] { 0.5, 0.5 }, s.getLayerActivations());
    }

}
