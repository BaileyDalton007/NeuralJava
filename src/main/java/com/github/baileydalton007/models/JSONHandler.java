package com.github.baileydalton007.models;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import com.github.baileydalton007.exceptions.ModelLoadingError;
import com.github.baileydalton007.models.components.BiasUnit;
import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.models.components.WeightMatrix;

/**
 * Class for handling JSON reading and writing for saving model configurations.
 * 
 * @author Bailey Dalton
 */
public class JSONHandler {
    /**
     * Takes the configuration of a model and writes them to a JSON file to saved to
     * disk and loaded later.
     * 
     * @param layerArray  The array of layers from the model to save
     * @param weightArray The array of weight matrices from the model to save
     * @param biasArray   The array of biases from the model to save
     * @param fileName    The name of the output JSON file
     * 
     * @throws IOException If the JSON file cannot be created in the current
     *                     directory to write the data to.
     */
    protected static void saveModelToJSONFile(Layer[] layerArray, WeightMatrix[] weightArray, BiasUnit[] biasArray,
            String fileName) throws IOException {
        // Creates an array to store the model layer JSON objects.
        JSONArray output = new JSONArray();

        // Interates through the layers in the model and adds them to the output.
        for (int i = 0; i < layerArray.length; i++) {
            // Stores the current layer.
            Layer layer = layerArray[i];

            // Adds the layer's size and activation to the JSON object.
            JSONObject jLayer = new JSONObject()
                    .put("size", layer.size())
                    .put("activation", layer.getActivationFunction().toString());

            // If the layer is not the input layer, add weights and biases to the JSON
            // object.
            if (i > 0) {
                // If the layer is not the
                jLayer
                        .put("weights", weightArray[i - 1].getMatrix())
                        .put("bias", biasArray[i - 1].getValue());
            }

            // Add the layer's JSON object to the output.
            output.put(jLayer);
        }

        // Write the output JSON array to a file with the name fileName.json.
        FileWriter file;

        // Will be the same fileName independent if the user puts .json or not.
        if (fileName.contains(".json"))
            file = new FileWriter(fileName);
        else
            file = new FileWriter(fileName + ".json");

        file.write(output.toString());
        file.close();

    }

    /**
     * Sets the inputted model's architecture and parameters based on the data
     * parsed from the input file.
     * 
     * @param fileName The name of the JSON file containing the model's data
     * @param model    The model being constructed
     * @throws FileNotFoundException If the input file cannot be found
     */
    protected static void loadJSONFile(String fileName, DenseNeuralNetwork model) throws FileNotFoundException {
        // Stores the scanner that will be used to read the input file.
        Scanner scanner;

        // Will add ".json" to the file name if the user did not input it.
        if (fileName.contains(".json"))
            scanner = new Scanner(new File(fileName));
        else
            scanner = new Scanner(new File(fileName + ".json"));

        // Iterates through each line of the input file and adds it to the string.
        StringBuilder string = new StringBuilder();
        while (scanner.hasNextLine()) {
            string.append(scanner.nextLine());
        }

        // Parses in file string to a JSON array storing the layer data.
        JSONArray layers = new JSONArray(string.toString());

        // Stores the layers that will be used to create the model.
        Layer[] layerArray = new Layer[layers.length()];

        // Stores the weight matrices that will be used to configure the model.
        WeightMatrix[] weightMatrixArray = new WeightMatrix[layerArray.length - 1];

        // Stores the biases that will be used to configure the model.
        BiasUnit[] biasArray = new BiasUnit[layerArray.length - 1];

        // Iterates through the JSON layer data, creating the layers and adding them to
        // the layer array.
        // Also stores the layer's weights and bias to be loaded.
        for (int layerIndex = 0; layerIndex < layers.length(); layerIndex++) {
            JSONObject layerData = layers.getJSONObject(layerIndex);

            layerArray[layerIndex] = new Layer(layerData.getInt("size"), layerData.getString("activation"));

            // Input layers don't weights or biases, so there is no need to parse them.
            if (layerIndex > 0) {
                // Sets the bias to the parsed value and adds it to the array.
                try {
                    biasArray[layerIndex - 1] = new BiasUnit(layerData.getDouble("bias"));
                } catch (Exception e) {
                    throw new ModelLoadingError(
                            "Check JSON data layer " + (layerIndex + 1)
                                    + ". Not enough biases in the model JSON. There should be one for every layer except the input layer. Error Message: "
                                    + e.getMessage());
                }

                // Stores the layer's weights in a JSONArray.
                JSONArray layerWeightsData = layerData.getJSONArray("weights");

                // If the weights for the layer from the data are less than expected, throw an
                // error.
                if (layerWeightsData.length() != layerArray[layerIndex].size())
                    throw new ModelLoadingError("Check JSON data layer " + (layerIndex + 1)
                            + ". Number of weights are different then expected from layer sizes. Expected "
                            + layerArray[layerIndex].size() + " but got " + layerWeightsData.length() + ".");

                // Stores all the layer's weights in a matrix
                double[][] layerWeights = new double[layerWeightsData.length()][];

                // Iterates through the rows of the matrix (the neurons in the current layer).
                for (int i = 0; i < layerWeightsData.length(); i++) {

                    // JSON array of the current neuron's weights.
                    JSONArray neuronWeightsData = layerWeightsData.getJSONArray(i);

                    // If the weights for a neuron from the data are less than expected, throw an
                    // error.
                    if (neuronWeightsData.length() != layerArray[layerIndex - 1].size())
                        throw new ModelLoadingError("Check JSON data layer " + (layerIndex + 1)
                                + ". Number of weights are different then expected from layer sizes. Expected "
                                + layerArray[layerIndex].size() + " but got " + neuronWeightsData.length() + ".");

                    // Stores the weights of the current neuron connecting to all the neurons in the
                    // previous layer.
                    double[] neuronWeights = new double[neuronWeightsData.length()];

                    // Iterates through each individual weight (each neuron connected in the
                    // previous layer).
                    for (int j = 0; j < neuronWeightsData.length(); j++) {
                        // Stores the weights in the current neuron matrix.
                        neuronWeights[j] = neuronWeightsData.getDouble(j);
                    }

                    // Stores the current neurons weights in the layer matrix.
                    layerWeights[i] = neuronWeights;
                }

                // Creates a weight matrix to store the weights of the current layer.
                WeightMatrix layerWeightMatrix = new WeightMatrix(layerWeights.length, layerWeights[0].length);
                layerWeightMatrix.setMatrix(layerWeights);

                // Stores the layer's weight matrix in the model's array of weight matrices.
                weightMatrixArray[layerIndex - 1] = layerWeightMatrix;
            }

        }

        // Sets the model's layer array to be the layers created from the JSON.
        model.setLayerArray(layerArray);

        // Sets the model's weight matrices to be the weights configured in the JSON.
        model.setWeightMatrices(weightMatrixArray);

        // Sets the model's bias array to be the biases configured in the JSON.
        model.setBiasArray(biasArray);

    }

}
