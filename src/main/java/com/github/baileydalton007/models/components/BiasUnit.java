package com.github.baileydalton007.models.components;

/**
 * Class for bias units for neural networks.
 * 
 * @author Bailey Dalton
 */
public class BiasUnit {
    // Stores the value of a bias unit.
    private double value;

    /**
     * Constructor for a bias unit, initializes its value to 1.0;
     */
    public BiasUnit() {
        value = 0.01;
    }

    /**
     * Getter for the bias unit's value.
     * 
     * @return The unit's value
     */
    public double getValue() {
        return value;
    }

    /**
     * Setter for the bias unit's value.
     * 
     * @param value The new value stored in the bias unit
     */
    public void setValue(double value) {
        this.value = value;
    }
}
