package com.github.baileydalton007.models;

import java.time.Duration;
import java.time.Instant;
import java.util.InputMismatchException;

import com.github.baileydalton007.activationfunctions.ActivationFunction;
import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.NetworkTooSmallException;
import com.github.baileydalton007.models.components.BiasUnit;
import com.github.baileydalton007.models.components.ErrorTensor;
import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.models.components.Neuron;
import com.github.baileydalton007.models.components.WeightMatrix;

/**
 * A class for Dense Neural Networks where every neuron in a layer is connected
 * to every neuron in the previous layer.
 * 
 * @author Bailey Dalton
 */
public class DenseNeuralNetwork {
    // Array for storing the layer objects in the network.
    private Layer[] layerArray;

    // Array for storing the weight matrix objects connecting each layer to the
    // previous in the network.
    private WeightMatrix[] layerWeights;

    // Array for storing each layer's bias unit.
    private BiasUnit[] layerBiases;

    // Gives field scope to the timing of training epochs.
    private Instant start;

    // The interval at which updates will be printed to the terminal during model
    // training.
    private int trainingUpdateInterval = 1;

    /**
     * Constructor for a dense neural network object.
     * 
     * @param layerArray An array of layers that will make up the network
     * @throws NetworkTooSmallException Will be thrown if a network is initialized
     *                                  with less than 2 layers
     */
    public DenseNeuralNetwork(Layer[] layerArray) throws NetworkTooSmallException {
        // Throw an exception if a network is created with less than two layers.
        if (layerArray.length < 2)
            throw new NetworkTooSmallException();

        this.layerArray = layerArray;

        // Weight matrix array will be the same size as the amount of layers minus one
        // as the input layer layer will not have a weight matrix.
        this.layerWeights = new WeightMatrix[layerArray.length - 1];

        // Bias array will be the same size as the amount of layers minus one
        // as the input layer layer will not have a bias.
        this.layerBiases = new BiasUnit[layerArray.length - 1];

        // Iterates through each layer and initializes an appropriately sized weight
        // matrix and bias.
        for (int i = 0; i < layerWeights.length; i++) {
            // Initializes weight matrices.
            layerWeights[i] = new WeightMatrix(layerArray[i + 1].size(), layerArray[i].size());

            // Initializes bias units.
            layerBiases[i] = new BiasUnit();
        }
    }

    /**
     * Training method for the dense neural network model. Will adjust
     * weights/biases to minmize error between the input and target values.
     * 
     * @param inputs       The training input values for the network. Each row is a
     *                     training example. Target outputs should map to the
     *                     targets array
     * @param targets      The target values for the network. Each row is a training
     *                     example. Training inputs should map to the inputs array
     * @param learningRate The amount which the weights are adjusted. Most often a
     *                     value between 0.0 and 1.0
     * @param epoch        Amount of cycles the network should train through the
     *                     training data
     * @param verbose      If true, data will be output for each
     *                     trainingUpdateInterval on the training process
     */
    public void train(double[][] inputs, double[][] targets, double learningRate, int epoch, boolean verbose) {
        // Stores the amount of digits in the input amount of epochs for padding console
        // output.
        int digitsInEpoch = (int) Math.log10(epoch) + 1;

        // Initializes the epoch timer to be displayed if verbose is enabled.
        this.start = Instant.now();

        // Cycle for each epoch of the training process.
        for (int i = 0; i < epoch; i++) {
            // Starts keeping track of the time elapsed to time the epoch.

            // Performs backpropagation and adjusts weights/biases accordingly relative to
            // the learning rate.
            this.backPropagation(inputs, targets, learningRate);

            // If in verbose mode, update the terminal on every interval.
            if (verbose) {
                printUpdateString(i, epoch, digitsInEpoch);
            }
        }

    }

    /**
     * Returns the model's output for a given inputs.
     * 
     * @param input The input array to the network, should be the same size as the
     *              input layer
     * @return The output of the network propagated from the input
     */
    public double[] predict(double[] input) {
        return forwardPropagation(input);
    }

    /**
     * Returns the model's output for an array of inputs.
     * 
     * @param input Array of inputs to the network. Each input array to the network
     *              should be the same size as the network's input layer
     * @return Array of the network's outputs propagated from the given inputs. Each
     *         row is a different output mapping to the same indexed input
     */
    public double[][] predict(double[][] input) {
        return forwardPropagation(input);
    }

    /**
     * Setter for the model's trainingUpdateInterval.
     * 
     * @param interval Amount of epochs between each update if training is set to
     *                 verbose.
     */
    public void setTrainingUpdateInterval(int interval) {
        if (interval < 1)
            throw new InputMismatchException("training update interval must be greater than or equal to 1");
        this.trainingUpdateInterval = interval;
    }

    /**
     * Getter for the model's trainingUpdateInterval.
     * 
     * @return interval Amount of epochs between each update if training is set to
     *         verbose.
     */
    public int getTrainingUpdateInterval() {
        return this.trainingUpdateInterval;
    }

    /**
     * Method that prints a training update to the terminal.
     * 
     * @param i             Current epoch being trained
     * @param epoch         The total amount of epochs that the model will be
     *                      trained
     * @param digitsInEpoch The amount of digits in the epoch integer
     */
    private void printUpdateString(int i, int epoch, int digitsInEpoch) {
        // Display an update if the current epoch is in the update interval, or it is
        // the first or last epoch.
        if ((i + 1) % trainingUpdateInterval == 0 || i == 0 || i == epoch - 1) {
            // Creates a string that will be output to the terminal.
            String updateString = new String();

            // Adds the epoch count.
            updateString = updateString.concat(String.format("| Epoch: %" + digitsInEpoch + "d | ", i + 1));

            // Adds the progress percentage.
            // If it is the last epoch, print "Done!" instead of percentage.
            if (i == epoch - 1)
                updateString = updateString.concat(String.format("Progress: Done! | "));
            else
                updateString = updateString
                        .concat(String.format("Progress: %04.1f%% | ", ((float) i / epoch) * 100));

            // Stores the duration of epoch.
            Duration duration = Duration.between(this.start, Instant.now());

            // Adds the duration of interval of epochs training.
            updateString = updateString
                    .concat(String.format("Time for %" + digitsInEpoch + "d Epochs: %dm %ds |", i + 1,
                            duration.getSeconds() / 60,
                            duration.getSeconds()));

            System.out.println(updateString);

            this.start = Instant.now();
        }
    }

    /**
     * Forward propagation algorithm calculating activations through the network.
     * 
     * @param input The input array to the network, should be the same size as the
     *              input layer
     * @return The output of the network propagated from the input
     * @throws IncompatibleInputException Thrown if the size of the input array is
     *                                    not the same as the input layer.
     */
    private double[] forwardPropagation(double[] input) throws IncompatibleInputException {
        // Checks that the input array is the same size as the input layer.
        if (input.length != layerArray[0].size())
            throw new IncompatibleInputException(
                    "Input size does not match size of first layer. Make sure that the input layer has the same amount of neurons as the input array.");

        // Feeds the input to the first layer.
        layerArray[0].input(input);

        // For each layer in the network, for each neuron in that layer, calculate the
        // activation by taking the weighted sum with all the previous layer's neurons.
        // Index starts at 1 since input layer does not need to be calculated.
        for (int layerIndex = 1; layerIndex < layerArray.length; layerIndex++) {

            // Stores the current layer being propagated.
            Layer currLayer = layerArray[layerIndex];

            // Stores the previous layer for calculating the weighted sum for the current
            // layer.
            Layer previousLayer = layerArray[layerIndex - 1];

            for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {

                // Stores the weighted sum that will become the input for the current neuron.
                double weightedSum = 0.0;

                // Stores the current neuron to which the input is being calculated.
                Neuron currNeuron = currLayer.getNeuron(neuronIndex);

                // Iterates through neurons in the previous layer, multiplies each value by its
                // weight, then adds to the sum total.
                for (int i = 0; i < previousLayer.size(); i++) {
                    weightedSum += previousLayer.getNeuron(i).getActivation() // The previous neuron's activation.
                            * this.layerWeights[layerIndex - 1].getMatrix()[neuronIndex][i]; // The weight connecting
                                                                                             // the current neuron to
                                                                                             // the previous neuron.
                }

                // Adds the bias to the weighted sum before being passed to the next neuron.
                weightedSum += layerBiases[layerIndex - 1].getValue();

                currNeuron.setInput(weightedSum);
            }

        }

        return getOutput();
    }

    /**
     * Forward propagation algorithm calculating activations through the network for
     * an array of inputs.
     * 
     * @param input Matrix of inputs to the network. Each row should be an input
     *              vector the size of the network's input layer
     * @return Matrix of the network's outputs. Each row is a different output
     *         mapping to the same indexed input
     * @throws IncompatibleInputException If the size of a training example does not
     *                                    match the size of the network's input
     *                                    layer
     */
    private double[][] forwardPropagation(double[][] input) throws IncompatibleInputException {
        // Creates a matrix to store the output.
        double[][] output = new double[input.length][];

        // Iterates through the inputs and propagates them before storing them in the
        // output.
        for (int i = 0; i < input.length; i++) {
            output[i] = forwardPropagation(input[i]);
        }

        return output;
    }

    /**
     * Gets the output of the network, the activations of the last layer in the
     * network.
     * 
     * @return An array of values representing the activations of the output layer
     */
    private double[] getOutput() {
        return layerArray[layerArray.length - 1].getLayerActivations();
    }

    /**
     * Implements the backpropagation algorithm for a dense neural network.
     * Adjusts weights an biases of network based on the input training examples and
     * errors in the network.
     * 
     * @param inputs       The input of the training examples
     * @param targets      The ground truth of the training examples, what the
     *                     network should output.
     * @param learningRate The amount which the weights are adjusted. Most often a
     *                     value between 0.0 and 1.0.
     */
    private void backPropagation(double[][] inputs, double[][] targets, double learningRate) {
        // Checks to make sure input and truth are the same size (number of rows).
        if (inputs.length != targets.length)
            throw new IncompatibleInputException(
                    "The amount of examples int the input should match the amount of examples int the expected answers.");

        // For neurons a error matrix is stored in the errorTensor.
        // For biases a error array is stored in the error Tensor
        ErrorTensor errorTensor = propagateErrorTensor(inputs, targets);

        // Unpacks the neuron error matrix from the error tensor.
        // The first index for layer L is (L-1) since the input layer is not included.
        double[][] neuronErrorMatrix = errorTensor.getNeuronErrors();

        // Unpacks the bias error array from the error tensor.
        // The first index for layer L is (L-1) since the input layer is not included.
        double[] biasErrorArray = errorTensor.getBiasErrors();

        // An array of weight matrices denoting the amount and direction which
        // each weight should be adjusted. Will be added to the network's array
        // of weight matrices to tune the network.
        WeightMatrix[] deltaWeights = getWeightAdjustments(learningRate, neuronErrorMatrix);

        // Iterates through each layer of weights, and updating them according to the
        // delta values.
        for (int layerIndex = 1; layerIndex < layerArray.length; layerIndex++) {
            this.layerWeights[layerIndex - 1].add(deltaWeights[layerIndex - 1]);
        }

        // Adjusts the model's biases for each layer based on the propagated error and
        // learning rate.
        adjustBiases(biasErrorArray, learningRate);
    }

    /**
     * Propagates the error for each neuron and bias in the network working
     * backwards.
     * 
     * For both the neuron error matrix and the bias error array, the first index
     * for layer L is (L-1) since the input layer is not included.
     * 
     * EX: The error value for the index 2 neuron on layer 4 would be:
     * errorTensor.getNeuronErrors[3][2]
     * 
     * EX: the bias error value for the second layer would be:
     * errorTensor.getBiasErrors[1]
     * 
     * @param inputs  The input values to the network that will be compared with the
     *                targets.
     * @param targets The expected value of the network.
     * @return An ErrorTensor which contains a 2D error matrix giving the error for
     *         each neuron in the network (excluding the input layer) and a 1D error
     *         array giving the error for each bias in the network (excluding the
     *         input layer).
     */
    private ErrorTensor propagateErrorTensor(double[][] inputs, double[][] targets) {
        // Creates a matrix to store the sum of all training example's errors for each
        // neuron.
        double[][] neuronErrorSum = new double[layerArray.length - 1][];

        // Creates an array to store the sum of all training example's errors for each
        // bias.
        double[] biasErrorSum = new double[layerArray.length - 1];

        // Iterates through all the training examples to calculate the sum error for the
        // training examples.
        for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {

            // Propagates the network forward with the given inputs to activate the
            // network's neurons.
            forwardPropagation(inputs[inputIndex]);

            // Creates a matrix to store each neuron's error.
            // Amount of errors - 1 since input layer is not included.
            double[][] neuronError = new double[layerArray.length - 1][];

            // Creates an array to store each layer's bias's error.
            // Amount of errors - 1 since input layer is not included.
            double[] biasError = new double[layerArray.length - 1];

            // Iterates backward through the netowrk's layers.
            // Does not include the input layer as the error is not needed.
            for (int layerIndex = layerArray.length - 1; layerIndex > 0; layerIndex--) {

                // Stores the current layer that will have its error calculated.
                Layer currLayer = layerArray[layerIndex];

                // Stores the current layer's activation function.
                ActivationFunction activationFunction = currLayer.getActivationFunction();

                // Creates a row in the matrices with the amount of columns needed to store the
                // error for each neuron in the layer.
                // layerIndex - 1 since input layer is not included.
                neuronError[layerIndex - 1] = new double[currLayer.size()];
                neuronErrorSum[layerIndex - 1] = new double[currLayer.size()];

                // Stores the activations for the current layer.
                double[] activations = currLayer.getLayerActivations();

                // Checks if the current layer is the output layer to use different error
                // calculation.
                if (layerIndex == layerArray.length - 1) {

                    // Iterates through neurons in current (output) layer to calculate error for
                    // each.
                    for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {
                        // ERROR = f'(WEIGHTED_SUM) * (TRUTH - ACTIVATION)
                        // where WEIGHTED_SUM is the input to the neuron, often denoted as the z term.
                        // where f'() is the derivative of the neuron's activation function.
                        // layerIndex - 1 since input layer is not included.
                        neuronErrorSum[layerIndex
                                - 1][neuronIndex] = neuronError[layerIndex - 1][neuronIndex] = (activationFunction
                                        .derivative(currLayer.getNeuron(neuronIndex).getInput()))
                                        * (targets[inputIndex][neuronIndex] - activations[neuronIndex]);

                        // BIAS_ERROR = SUM_FOR_NEURONS_IN_LAYER(f'(BIAS_VALUE) * (TRUTH - ACTIVATION))
                        // where f'() is the derivative of the layer's activation function.
                        // layerIndex - 1 since input layer does not have a bias.
                        biasErrorSum[layerIndex - 1] = biasError[layerIndex - 1] += activationFunction
                                .derivative(layerBiases[layerIndex - 1].getValue())
                                * (targets[inputIndex][neuronIndex] - activations[neuronIndex]);
                    }
                } else {
                    // If this layer is not an output layer.

                    // Iterates through neurons in current (hidden) layer to calculate error for
                    // each.
                    for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {
                        // Stores the sum of the errors and weights in the next layer.
                        double errSum = 0.0;

                        // Iterates through each neuron, adding the product of the error and the weight
                        // connecting it to the current neuron to the sum.
                        for (int k = 0; k < layerArray[layerIndex + 1].size(); k++) {
                            // layerIndex + 1 - 1 for the next layer's errors since input layer is not
                            // included (for both error and weight matrices).
                            errSum += neuronError[layerIndex][k] * layerWeights[layerIndex].getMatrix()[k][neuronIndex];
                        }

                        // ERROR = f'(WEIGHTED_SUM) * (ERROR_SUM)
                        // where WEIGHTED_SUM is the input to the neuron, often denoted as the z term.
                        // Where f'() is the derivative of the neuron's activation function.
                        // And ERROR_SUM is the sum of the products of all the errors and weights in the
                        // next layer.
                        // layerIndex - 1 since input layer is not included.
                        neuronError[layerIndex - 1][neuronIndex] = activationFunction
                                .derivative(currLayer.getNeuron(neuronIndex).getInput()) * errSum;

                        // Adds the above neuron's error to the error sum array.
                        neuronErrorSum[layerIndex - 1][neuronIndex] += neuronError[layerIndex - 1][neuronIndex];

                        // BIAS_ERROR = SUM_FOR_NEURONS_IN_LAYER(f'(BIAS_VALUE) * NEURON_ERROR)
                        // where f'() is the derivative of the layer's activation function.
                        // layerIndex - 1 since input layer does not have a bias.
                        biasError[layerIndex - 1] += activationFunction
                                .derivative(layerBiases[layerIndex - 1].getValue())
                                * neuronError[layerIndex - 1][neuronIndex];

                        // Adds the above bias unit's error to the error sum array.
                        biasErrorSum[layerIndex - 1] += biasError[layerIndex - 1];
                    }
                }
            }
        }

        // Divide the sums of each bias's error by the number of inputs.
        // This array is the average error for each bias in the network.
        for (int i = 0; i < biasErrorSum.length; i++) {
            biasErrorSum[i] = biasErrorSum[i] / inputs.length;
        }

        // Divide the sums of each neuron's error by the number of inputs.
        // This matrix is the average error for each neuron in the network.
        for (int i = 0; i < neuronErrorSum.length; i++) {
            for (int j = 0; j < neuronErrorSum[i].length; j++) {
                neuronErrorSum[i][j] = neuronErrorSum[i][j] / inputs.length;
            }
        }

        // Pack the neuron average error matrix and the bias average error array into an
        // error tensor.
        return new ErrorTensor(neuronErrorSum, biasErrorSum);
    }

    /**
     * Uses the error for each neuron in the network to calculate the appropriate
     * change to make to each weight, and returns a array of (delta) weight matrices
     * that can be added to the network's current array of weight matrices.
     * 
     * @param learningRate The relativea mount which the network will adjust
     *                     weights. Suggested to be around 1.0.
     * @param errorMatrix  The matrix denoting the error in each neuron, calculated
     *                     from propagateErrorMatrix.
     * @return An array of weight matrices denoting the amount and direction which
     *         each weight should be adjusted. Will be added to the network's array
     *         of weight matrices to tune the network.
     */
    private WeightMatrix[] getWeightAdjustments(double learningRate, double[][] errorMatrix) {
        // Array to store the changes to each weight calculated from the errors.
        WeightMatrix[] deltaWeights = new WeightMatrix[layerArray.length - 1];

        // Iterates backward through the netowrk's layers.
        // Does not include the input layer as the error is not needed.
        for (int layerIndex = layerArray.length - 1; layerIndex > 0; layerIndex--) {
            // Stores the current layer.
            Layer currLayer = layerArray[layerIndex];

            // Stores the previous layer.
            Layer prevLayer = layerArray[layerIndex - 1];

            // Creates a weight matrix that will store this layer's changes to the weights.
            deltaWeights[layerIndex - 1] = new WeightMatrix(currLayer.size(), prevLayer.size());

            // A 2D array that will store the changes to the weights before adding them to
            // the weight matrix.
            double[][] deltaW = new double[currLayer.size()][prevLayer.size()];

            // Stores the activations of the previous layer.
            double[] prevActivations = prevLayer.getLayerActivations();

            // Iterates through each neuron in the current layer.
            for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {

                // Iterates through each neuron in the previous layer.
                // Calculates the change that should be applied to the weight connecting the
                // current neuron to the previous neuron (k-th neuron), and stores it in the 2D
                // array.
                for (int k = 0; k < prevLayer.size(); k++) {
                    // DELTA_W = LEARNING_RATE * (CURR_NEURON_ERROR) *
                    // (ACTIVATION_OF_PREVIOUS_NEURON)
                    deltaW[neuronIndex][k] = learningRate * errorMatrix[layerIndex - 1][neuronIndex]
                            * prevActivations[k];
                }
            }

            // Sets the layer's delta weight matrix to the 2D array storing the delta w
            // values.
            deltaWeights[layerIndex - 1].setMatrix(deltaW);
        }

        return deltaWeights;
    }

    /**
     * Adjusts each bias in the model due to the bias error array passed into it.
     * 
     * @param errorArray   Bias error array propagated during backpropagation
     * @param learningRate The model's learning rate, relative amount that the bias
     *                     will be adjusted
     */
    private void adjustBiases(double[] errorArray, double learningRate) {
        // Iterates through each bias to update them.
        for (int i = 0; i < errorArray.length; i++) {
            // Stores the current bias unit.
            BiasUnit currBias = layerBiases[i];

            // b = b + LEARNING_RATE * BIAS_ERROR
            // Updates the current bias based on the learning rate and that bias's error.
            currBias.setValue(currBias.getValue() + learningRate * errorArray[i]);
        }
    }
}
