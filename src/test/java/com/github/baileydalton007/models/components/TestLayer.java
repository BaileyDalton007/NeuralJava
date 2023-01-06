package com.github.baileydalton007.models.components;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

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
    public void TestSize() {
        assertEquals(1, new Layer(1, "relu"));
    }
}
