/**
 * Class for layers that make up neural networks.
 * 
 * @author Bailey Dalton
 */
public class Layer {
    // Array storing the layer's neurons.
    private Neuron[] neurons;

    /**
     * Constructor for a layer instance.
     * 
     * @param numNeurons         The number of neurons that should make up the
     *                           layer
     * @param activationFunction The activation function that each neuron in the
     *                           layer should use.
     */
    public Layer(int numNeurons, ActivationFunction activationFunction) {
        // Creates an array to store the neurons in the layer.
        neurons = new Neuron[numNeurons];

        // Creates and stores neurons in the array.
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(activationFunction);
        }
    }

    public double[] getLayerActivations() {
        // Creates an array to store activations for the neurons in the layer.
        double[] activations = new double[neurons.length];

        // Iterates through neurons to calculate and store the activation for each.
        for (int i = 0; i < neurons.length; i++) {
            activations[i] = neurons[i].getActivation();
        }

        return activations;
    }
}
