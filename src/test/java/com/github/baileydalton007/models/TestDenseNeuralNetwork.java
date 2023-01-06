package com.github.baileydalton007.models;

import org.junit.jupiter.api.Test;

import com.github.baileydalton007.models.components.Layer;

/**
 * Unit test for testing the dense neural network model.
 * 
 * @author Bailey Dalton
 */
public class TestDenseNeuralNetwork {

    @Test
    public void test() {
        DenseNeuralNetwork model = new DenseNeuralNetwork(new Layer[] {
                new Layer(3, "relu"),
                new Layer(2, "relu"),
                new Layer(1, "sigmoid"),
        });

    }
}
