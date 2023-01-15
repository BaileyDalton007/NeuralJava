package com.github.baileydalton007.models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.InputMismatchException;

import com.github.baileydalton007.activationfunctions.ActivationFunction;
import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.ModelLoadingError;
import com.github.baileydalton007.exceptions.NetworkTooSmallException;
import com.github.baileydalton007.lossFunctions.SumSquareResidual;
import com.github.baileydalton007.models.components.BiasUnit;
import com.github.baileydalton007.models.components.ErrorTensor;
import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.models.components.Neuron;
import com.github.baileydalton007.models.components.WeightMatrix;
import com.github.baileydalton007.utils.TensorOperations;
import com.github.baileydalton007.utils.TextBox;

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

    // The model's current loss while training.
    private double trainMeanSquaredError;

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
     * Loads a model configuration from a JSON file.
     * 
     * @param fileName The name of the JSON file to load the model from
     */
    public DenseNeuralNetwork(String fileName) {
        try {
            // Handler will parse the data then use this model's setters to externally set
            // the weights, layers, and biases.
            JSONHandler.loadJSONFile(fileName, this);
        } catch (ModelLoadingError e) {
            throw e;
        } catch (FileNotFoundException e) {
            throw new ModelLoadingError(
                    "Model file could not be found in this directory. Error Message: " + e.getMessage());

        } catch (Exception e) {
            throw new ModelLoadingError("Error loading model. Error Message: " + e.getMessage());
        }
    };

    /**
     * Overrides the model's toString method to return information on the model's
     * architecture.
     * 
     * @return String describing the network and its layers.
     */
    @Override
    public String toString() {
        // Stores the strings containing layer information
        StringBuilder layerStrings = new StringBuilder();

        // Stores the total number of weights in the network.
        int numWeights = 0;
        for (int i = 0; i < layerArray.length; i++) {
            // Add the number of neurons in this layer multiplied by the number of neurons
            // in the previous layer, giving the number of weights between the two layers.
            // Checks if it is the first layer as it does not have any weights.
            if (i > 0)
                numWeights += layerArray[i].size() * layerArray[i - 1].size();

            // Determines the current layer's type to be added to the string.
            String layerType;
            if (i == 0)
                layerType = "input";
            else if (i == layerArray.length - 1)
                layerType = "output";
            else
                layerType = "hidden";

            // Adds this layer's string to the layerStrings string containing all layers.
            layerStrings.append(String.format("| Type: %7s " + layerArray[i].toString() + "\n", layerType));
        }

        // Array of each line in the model information section that will be passed into
        // the TextBox object to be padded and formatted.
        String[] strArr = new String[] {
                String.format("Model Information"),
                String.format("Type: Dense Neural Network"),
                String.format("Input Size: %d", layerArray[0].size()),
                String.format("Output Size: %d", layerArray[layerArray.length - 1].size()),
                String.format("Number of Layers: %d", layerArray.length),
                String.format("Total Weights: %d", numWeights),
                String.format("Total Biases: %d", layerBiases.length),
                String.format("Layer Information")
        };

        // Stores the length of each line in the layer string.
        // Every variable is padded so should stay constant;
        final int LAYER_STRING_LENGTH = 68;

        // TextBox object for formatting text and adding borders.
        TextBox boxObject = new TextBox(strArr, LAYER_STRING_LENGTH);

        // Creates a string builder to store the output.
        StringBuilder output = new StringBuilder();

        // Adds the First title (Model Information).
        output.append(boxObject.getTopBox());
        output.append(boxObject.getText(0));
        output.append(boxObject.getBottom());

        // Adds the model information.
        output.append(boxObject.getText(1));
        output.append(boxObject.getText(2));
        output.append(boxObject.getText(3));
        output.append(boxObject.getText(4));
        output.append(boxObject.getText(5));
        output.append(boxObject.getText(6));

        // Adds the second title (Layer Information)
        output.append(boxObject.getBottom());
        output.append(boxObject.getText(7));
        output.append(boxObject.getBottom());

        // Adds the layer info to output.
        output.append(layerStrings);
        output.append(boxObject.getDividerBox(16, 42));

        return output.toString();
    }

    /**
     * Training method for the dense neural network model. Will adjust
     * weights/biases to minmize error between the input and target values.
     * 
     * @param inputs       The training input values for the network. Each row is a
     *                     training example. Target outputs should map to the
     *                     targets array
     * @param targets      The target values for the network. Each row is a training
     *                     example. Training targets should map to the inputs array
     * @param testInputs   Test input values for the network. Will be used to
     *                     calculate the model's test error. Each row is a test
     *                     example. Test inputs should map to the testTargets array
     * @param testTargets  Test target values for the network. Will be used to
     *                     calculate the model's test error. Test targets should map
     *                     to the test inputs array
     * @param learningRate The amount which the weights are adjusted. Most often a
     *                     value between 0.0 and 1.0
     * @param epoch        Amount of cycles the network should train through the
     *                     training data
     * @param verbose      If true, data will be output for each
     *                     trainingUpdateInterval on the training process
     */
    public void train(double[][] inputs, double[][] targets, double[][] testInputs, double[][] testTargets,
            double learningRate, int epoch, boolean verbose) {
        // Stores the amount of digits in the input amount of epochs for padding console
        // output.
        int digitsInEpoch = (int) Math.log10(epoch) + 1;

        // Initializes the epoch timer to be displayed if verbose is enabled.
        this.start = Instant.now();

        // Initializes the time for the whole training process, will not be updated.
        Instant begin = Instant.now();

        // Cycle for each epoch of the training process.
        for (int i = 0; i < epoch; i++) {
            // Starts keeping track of the time elapsed to time the epoch.

            // Performs backpropagation and adjusts weights/biases accordingly relative to
            // the learning rate.
            this.backPropagation(inputs, targets, learningRate);

            // Stores the test error of the network in its current point in training with
            // the passed in test dataset.
            double testError = this.benchmarkError(testInputs, testTargets);

            // If in verbose mode, update the terminal on every interval.
            if (verbose) {
                printUpdateString(i, epoch, digitsInEpoch, testError, begin);
            }
        }

    }

    /**
     * Returns the model's output for a given inputs.
     * 
     * @param input The input array (or single value) to the network, should be the
     *              same size as the input layer
     * @return The output of the network propagated from the input
     */
    public double[] predict(double... input) {
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
     * Returns a string that describes the model's architecture.
     * Call's the model's toString() method.
     * 
     * @return String describing the model.
     */
    public String info() {
        return this.toString();
    }

    /**
     * Saves the model's architecture and weight's / biases to a JSON file.
     * 
     * @param fileName The name of the output JSON file.
     */
    public void saveToFile(String fileName) {
        try {
            JSONHandler.saveModelToJSONFile(layerArray, layerWeights, layerBiases, fileName);
        } catch (IOException e) {
            throw new ModelLoadingError(
                    "Could not create the JSON file in the current directory. Error Message: " + e.getMessage());
        }
    }

    /**
     * Method that prints a training update to the terminal.
     * 
     * @param i             Current epoch index being trained
     * @param epoch         The total amount of epochs that the model will be
     *                      trained
     * @param digitsInEpoch The amount of digits in the epoch integer
     * @param testError     The calculated error of the network with a test dataset.
     * @param begin         The instant training started, will be used to calculate
     *                      total time elapsed.
     */
    private void printUpdateString(int i, int epoch, int digitsInEpoch, double testError, Instant begin) {
        // Display an update if the current epoch is in the update interval, or it is
        // the first or last epoch.
        if ((i + 1) % trainingUpdateInterval == 0 || i == 0 || i == 99 || i == epoch - 1) {
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
                        .concat(String.format("Progress: %04.1f%% | ", ((float) (i + 1) / epoch) * 100));

            // Stores the duration of epoch.
            Instant now = Instant.now();
            Duration duration = Duration.between(this.start, now);

            // Adds the duration of interval of epochs training.
            updateString = updateString
                    .concat(String.format("Time Spent: %dm %2ds | ",
                            duration.getSeconds() / 60,
                            duration.getSeconds() % 60));

            // Adds average training error.
            updateString = updateString.concat(String.format("Train Error: %10.3f | ", this.trainMeanSquaredError));

            // Adds test error.
            updateString = updateString.concat(String.format("Test Error: %10.3f | ", testError));

            // Uses the time spent training each epoch in this interval to estimate how long
            // the rest of training will take.
            // Some algebraic rearranging has been done to minimize the amount of operations
            // done on Duration.
            Duration timeRemaining = duration.multipliedBy((epoch - (i + 1)) / trainingUpdateInterval);

            // Adds the estimated time remaining in training.
            updateString = updateString
                    .concat(String.format("Est. Remaining: %dm %2ds | ",
                            timeRemaining.getSeconds() / 60,
                            timeRemaining.getSeconds() % 60));

            // The time elapsed from starting training to now.
            Duration timeElapsed = Duration.between(begin, now);

            // Adds the time elapsed during training.
            updateString = updateString
                    .concat(String.format("Elapsed: %dm %2ds | ",
                            timeElapsed.getSeconds() / 60,
                            timeElapsed.getSeconds() % 60));

            System.out.println(updateString);

            // Updates the time interval for the next loop.
            this.start = Instant.now();
        }
    }

    /**
     * A method to determine the network's Mean Squared Error (MSE) based on a test
     * or validation dataset.
     * 
     * @param testInputs  Array of test or validation inputs to the network. Each
     *                    input array to the
     *                    network should be the same size as the network's input
     *                    layer.
     * @param testTargets Array of test or validation targets for the network. Each
     *                    input array to the
     *                    network should be the same size as the network's output
     *                    layer.
     * @return Mean Squared Error (MSE) of the network over the test/validation
     *         dataset.
     */
    private double benchmarkError(double[][] testInputs, double[][] testTargets) {
        // Stores the sum of the errors in all the input / target pairs.
        double errorSum = 0.0;

        // Propagates the network's outputs from the test inputs.
        double[][] outputs = forwardPropagation(testInputs);

        // Interates through each input / target pair and calculates the sum of their
        // errors to be averaged.
        for (int exampleIndex = 0; exampleIndex < outputs.length; exampleIndex++) {

            // Stores the error sum for each test example.
            double exampleErrorSum = 0.0;

            // Iterates through all the values in the output (neurons in the output layer),
            // and sums up their MSE.
            for (int activationIndex = 0; activationIndex < outputs[exampleIndex].length; activationIndex++) {
                exampleErrorSum += Math
                        .pow(outputs[exampleIndex][activationIndex] - testTargets[exampleIndex][activationIndex], 2);
            }

            // Adds the test example's MSE to the network's error sum.
            errorSum += exampleErrorSum;

        }

        // Averages the errors in the network to return the network's Mean Square Error
        // (MSE).
        return errorSum / testInputs.length;
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
        layerArray[0].updateLayerActivations();

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
                    weightedSum += previousLayer.getLayerActivations()[i] // The previous neuron's activation.
                            * this.layerWeights[layerIndex - 1].getMatrix()[neuronIndex][i]; // The weight connecting
                                                                                             // the current neuron to
                                                                                             // the previous neuron.
                }

                // Adds the bias to the weighted sum before being passed to the next neuron.
                weightedSum += layerBiases[layerIndex - 1].getValue();

                currNeuron.setInput(weightedSum);
            }

            // Stores the layer's activations in the layer object for optimized access
            // later.
            currLayer.updateLayerActivations();

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
                    "The amount of examples in the input should match the amount of examples int the expected answers.");

        // For neurons a error matrix is stored in the errorTensor.
        // For biases a error array is stored in the error Tensor
        ErrorTensor errorTensor = propagateErrorTensor(inputs, targets);

        // Unpacks the neuron error matrix from the error tensor.
        // The first index for layer L is (L-1) since the input layer is not included.
        double[][][] weightErrors = errorTensor.getWeightErrors();

        // Unpacks the bias error array from the error tensor.
        // The first index for layer L is (L-1) since the input layer is not included.
        double[] biasErrorArray = errorTensor.getBiasErrors();

        // Iterates through each layer adjusting the weights according to the errors.
        for (int layerIndex = 1; layerIndex < layerArray.length; layerIndex++) {
            double[][] currWeights = this.layerWeights[layerIndex - 1].getMatrix();
            double[][] deltaWeights = TensorOperations.multiplyByValue(weightErrors[layerIndex - 1], learningRate);

            this.layerWeights[layerIndex - 1].setMatrix(TensorOperations.subtractElements(currWeights, deltaWeights));
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
        // Creates a 3D tensor that will hold the sum of all the errors for each weight.
        double[][][] weightErrorSums = new double[layerArray.length - 1][][];

        // Creates an array that will store the sums of all errors for each bias.
        double[] biasErrorSums = new double[layerArray.length - 1];

        // Initializes the weightErrorSums tensor as the same size as each weight matrix
        // in each layer.
        for (int i = 1; i < layerArray.length; i++) {
            weightErrorSums[i - 1] = new double[layerArray[i].size()][layerArray[i - 1].size()];
        }

        // Iterates through each input example, and sums up the error for all of them.
        for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            forwardPropagation(inputs[inputIndex]);

            // Will store the errors in the previous iteration (the layer in front of this
            // one because we iterate backward).
            double[] prevIterationErrors = new double[0];

            // Iterates through each layer backward propagating error.
            for (int layerIndex = layerArray.length - 1; layerIndex > 0; layerIndex--) {
                Layer currLayer = layerArray[layerIndex];
                Layer prevLayer = layerArray[layerIndex - 1];

                // Stores the activations for the previous layer to be used to calculate each
                // weight's error.
                double[] prevActivations = prevLayer.getLayerActivations();

                // Array that will store the error in each neuron.
                double[] neuronError = new double[currLayer.size()];

                // If this layer is the network's output layer.
                if (layerIndex == layerArray.length - 1) {

                    // Stores the sum for the layer's error to be used to calculate the model's
                    // training error.
                    double meanSquaredErrorSum = 0.0;

                    // MSE = SUM(|ai - yi| ^ 2) / NEURONS_IN_OUTPUT
                    // Mean squared error calculation.
                    // Where ai is the activation of output neuron i.
                    // Where yi is the target activation of output neuron i.
                    meanSquaredErrorSum += Math
                            .pow(TensorOperations.sumElements(currLayer.getLayerActivations())
                                    - TensorOperations.sumElements(targets[inputIndex]), 2);

                    // Averages the training mean squared error for the output layer.
                    this.trainMeanSquaredError = meanSquaredErrorSum / currLayer.size();

                    // Calculates the output layer's loss.
                    neuronError = SumSquareResidual.loss(currLayer, null, targets[inputIndex], true);

                    // Stores this error for calculations in the next iteration.
                    prevIterationErrors = neuronError;

                    // Calculates the error the bias in this layer.
                    biasErrorSums[layerIndex - 1] = SumSquareResidual.lossBias(neuronError,
                            layerBiases[layerIndex - 1].getValue(), targets[inputIndex], currLayer, true);

                } else {
                    // Will Store the sum of errors from the next layer.
                    double[] nextLayerErrorSums = new double[currLayer.size()];

                    // Iterates through the neurons in the current error getting their error sums.
                    for (int i = 0; i < currLayer.size(); i++) {
                        // Iterates through the neurons in the next layer adding up their errors
                        // multiplied by the weight connecting them to the current neuron.
                        for (int j = 0; j < layerArray[layerIndex + 1].size(); j++) {
                            nextLayerErrorSums[i] += prevIterationErrors[j]
                                    * layerWeights[layerIndex].getMatrix()[j][i];
                        }
                    }

                    // Calculates this hidden layer's loss.
                    neuronError = SumSquareResidual.loss(currLayer, nextLayerErrorSums, targets[inputIndex], false);

                    // Stores this error for calculations in the next iteration.
                    prevIterationErrors = neuronError;

                    // Calculates the error the bias in this layer.
                    biasErrorSums[layerIndex - 1] = SumSquareResidual.lossBias(neuronError,
                            layerBiases[layerIndex - 1].getValue(), targets[inputIndex], currLayer, false);
                }

                // Gets the influnce of the previous layer's neurons to calculate the error in
                // the weights that connect the previous layer to this layer.
                double[][] previousLayerInfluence = previousLayerInfluence(currLayer.size(), prevActivations,
                        layerWeights[layerIndex - 1].getMatrix());

                // Multiplies each weight's influence by its neuron's error.
                double[][] weightErr = TensorOperations.multiplyRowsByVector(neuronError, previousLayerInfluence);

                // Adds this input's weight errors to the summation to be averaged over all
                // inputs.
                weightErrorSums[layerIndex - 1] = TensorOperations.addElements(weightErrorSums[layerIndex - 1],
                        weightErr);
            }
        }

        // Averages the weight errors over all inputs.
        weightErrorSums = TensorOperations.divideByValue(weightErrorSums, inputs.length);

        return new ErrorTensor(weightErrorSums, biasErrorSums);
    }

    /**
     * Multiplies each of the activations in the previous layer by the weights that
     * connects them to the current layer.
     * 
     * @param currNumNeurons      The number of neurons in the current layer
     * @param previousActivations The activations of the previous layer
     * @param weights             The weights that connect this layer to the
     *                            previous layer
     * @return A matrix the size of the layer's weight matrix where each value is
     *         the neuron's from the previous layer's activation multiplied by the
     *         weight connecting the previous neuron to the current neuron.
     */
    private double[][] previousLayerInfluence(int currNumNeurons, double[] previousActivations, double[][] weights) {
        double[][] output = new double[currNumNeurons][];

        // Interates through each neuron in the current layer.
        for (int currNeuronIndex = 0; currNeuronIndex < currNumNeurons; currNeuronIndex++) {
            // Creates a row for each neuron in the current layer.
            output[currNeuronIndex] = new double[previousActivations.length];

            // Iterates through each neuron in the previous layer
            for (int prevNeuronIndex = 0; prevNeuronIndex < previousActivations.length; prevNeuronIndex++) {

                // Multiplies the previous neuron's activation by the weight that connects it to
                // this layer and stores it in the matrix.
                output[currNeuronIndex][prevNeuronIndex] = previousActivations[prevNeuronIndex]
                        * weights[currNeuronIndex][prevNeuronIndex];
            }
        }

        return output;
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

    /**
     * Setter for the model's layer array.
     * 
     * @param The layer array to set the model's layer array to.
     */
    protected void setLayerArray(Layer[] layerArray) {
        this.layerArray = layerArray;
    }

    /**
     * Setter for the model's weight matrices.
     * 
     * @param The array of weight matrices to set the model's weight matrices to.
     */
    protected void setWeightMatrices(WeightMatrix[] weightMatrixArray) {
        this.layerWeights = weightMatrixArray;
    }

    /**
     * Setter for the model's bias array.
     * 
     * @param The bias array to set the model's bias array to.
     */
    protected void setBiasArray(BiasUnit[] biasArray) {
        this.layerBiases = biasArray;
    }
}
