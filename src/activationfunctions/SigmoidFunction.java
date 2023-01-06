package activationfunctions;

/**
 * Sigmoid activation function for deep learning neurons.
 * 
 * @author Bailey Dalton
 */
public class SigmoidFunction extends ActivationFunction {

    /**
     * Method to apply the sigmoid function to an input.
     * 
     * @param x Input to the sigmoid function
     * @return The output of the sigmoid function with x as an input.
     */
    @Override
    public Double apply(Double x) {
        return 1 / (1 + Math.exp(-x));
    }

}
