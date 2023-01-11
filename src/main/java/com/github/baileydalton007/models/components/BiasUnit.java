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
     * Constructor for a bias unit, initializes its value to 0.1;
     */
    public BiasUnit() {
        value = 0.1;
    }

    /**
     * Constructor for a bias unit, initializes its value to the input value;
     * 
     * @param value The value that the bias should hold
     */
    public BiasUnit(double value) {
        this.value = value;
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
