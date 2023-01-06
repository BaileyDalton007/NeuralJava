package com.github.baileydalton007.models;

import org.junit.jupiter.api.Test;

import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.activationfunctions.*;;

/**
 * Unit test for testing the dense neural network model.
 * 
 * @author Bailey Dalton
 */
public class TestDenseNeuralNetwork {

    @Test
    public void test() {
        DenseNeuralNetwork model = new DenseNeuralNetwork(new Layer[] {
                new Layer(3, new ReLUFunction()),
                new Layer(2, new ReLUFunction()),
                new Layer(1, new SigmoidFunction()),
        });

    }
}
