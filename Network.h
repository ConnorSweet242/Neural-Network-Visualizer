#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <functional>

// ============================================================
//  Network.h
//  A small, dependency-free feedforward neural network.
//  No linear algebra library -- plain loops, so every step
//  (weights, sums, activations, gradients) is easy to trace
//  and hook into the visualization layer.
// ============================================================

// Sigmoid activation and its derivative (derivative expressed
// in terms of the *output* of sigmoid, which is the common
// convenient form: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)))
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoidDerivativeFromOutput(double sigmoidOutput) {
    return sigmoidOutput * (1.0 - sigmoidOutput);
}

// ------------------------------------------------------------
// Neuron
// Holds the values the visualization needs every frame:
//   - activation: the value shown as the neuron's fill color
//   - weightedSum: pre-activation value (z), kept for backprop
//   - delta: backprop error signal at this neuron (Phase 6)
//   - incomingWeights / weightGradients: one per input connection
//   - bias / biasGradient
// ------------------------------------------------------------
struct Neuron {
    double activation = 0.0;
    double weightedSum = 0.0;
    double delta = 0.0;

    double bias = 0.0;
    double biasGradient = 0.0;

    std::vector<double> incomingWeights;   // weights[i] connects prevLayer.neuron[i] -> this neuron
    std::vector<double> weightGradients;   // same shape as incomingWeights, filled during backprop

    void initWeights(int numInputs, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        incomingWeights.resize(numInputs);
        weightGradients.assign(numInputs, 0.0);
        for (double& w : incomingWeights) w = dist(rng);
        bias = dist(rng) * 0.5;
    }
};

// ------------------------------------------------------------
// Layer
// A simple container of neurons. The input layer is a layer
// with no incoming weights (its "activation" is just set
// directly from external input data).
// ------------------------------------------------------------
struct Layer {
    std::vector<Neuron> neurons;

    explicit Layer(int size) : neurons(size) {}

    int size() const { return static_cast<int>(neurons.size()); }
};

// ------------------------------------------------------------
// Network
// Holds layers[0] = input layer ... layers[N-1] = output layer.
// Provides:
//   forward(inputs)            -- Phase 5
//   backward(targets, lr)      -- Phase 6
// Both are written step-by-step (not vectorized) so each line
// maps to one thing the visualizer can highlight/animate.
// ------------------------------------------------------------
class Network {
public:
    // layerSizes example: {2, 4, 4, 1} -> 2 inputs, two hidden layers of 4, 1 output
    Network(const std::vector<int>& layerSizes, unsigned seed = 42)
        : rng(seed)
    {
        for (int size : layerSizes) {
            layers.emplace_back(size);
        }
        // Initialize weights for every layer except the input layer.
        for (size_t l = 1; l < layers.size(); ++l) {
            int prevSize = layers[l - 1].size();
            for (Neuron& n : layers[l].neurons) {
                n.initWeights(prevSize, rng);
            }
        }
    }

    int numLayers() const { return static_cast<int>(layers.size()); }
    Layer& layer(int index) { return layers[static_cast<size_t>(index)]; }
    const Layer& layer(int index) const { return layers[static_cast<size_t>(index)]; }

    // ---------------- Phase 5: Forward propagation ----------------
    // Sets input layer activations directly, then for every
    // subsequent layer computes weightedSum and activation.
    std::vector<double> forward(const std::vector<double>& inputs) {
        // 1. Load inputs into the input layer
        Layer& inputLayer = layers.front();
        for (int i = 0; i < inputLayer.size(); ++i) {
            inputLayer.neurons[static_cast<size_t>(i)].activation = inputs[static_cast<size_t>(i)];
            inputLayer.neurons[static_cast<size_t>(i)].weightedSum = inputs[static_cast<size_t>(i)];
        }

        // 2. Propagate through each subsequent layer
        for (size_t l = 1; l < layers.size(); ++l) {
            Layer& prev = layers[l - 1];
            Layer& curr = layers[l];

            for (Neuron& n : curr.neurons) {
                double sum = n.bias;
                for (size_t i = 0; i < n.incomingWeights.size(); ++i) {
                    sum += n.incomingWeights[i] * prev.neurons[i].activation;
                }
                n.weightedSum = sum;
                n.activation = sigmoid(sum);
            }
        }

        // 3. Return output layer activations
        std::vector<double> out;
        for (Neuron& n : layers.back().neurons) out.push_back(n.activation);
        return out;
    }

    // ---------------- Phase 6: Backpropagation ----------------
    // targets: desired output values (same size as output layer)
    // learningRate: step size for gradient descent
    // Returns the mean-squared error loss for this pass (for display).
    double backward(const std::vector<double>& targets, double learningRate) {
        // 1. Compute delta (error signal) for the output layer:
        //    delta = (activation - target) * sigmoid'(activation)
        Layer& outputLayer = layers.back();
        double loss = 0.0;
        for (size_t i = 0; i < outputLayer.neurons.size(); ++i) {
            Neuron& n = outputLayer.neurons[i];
            double error = n.activation - targets[i];
            loss += error * error;
            n.delta = error * sigmoidDerivativeFromOutput(n.activation);
        }
        loss /= static_cast<double>(outputLayer.neurons.size());

        // 2. Propagate delta backward through hidden layers.
        //    For a hidden neuron, delta = (sum over next-layer neurons of
        //    nextNeuron.delta * weight connecting this neuron to it) * sigmoid'(activation)
        for (int l = static_cast<int>(layers.size()) - 2; l >= 1; --l) {
            Layer& curr = layers[static_cast<size_t>(l)];
            Layer& next = layers[static_cast<size_t>(l + 1)];

            for (size_t i = 0; i < curr.neurons.size(); ++i) {
                double sum = 0.0;
                for (Neuron& nextNeuron : next.neurons) {
                    sum += nextNeuron.delta * nextNeuron.incomingWeights[i];
                }
                curr.neurons[i].delta = sum * sigmoidDerivativeFromOutput(curr.neurons[i].activation);
            }
        }

        // 3. Compute gradients and update weights/biases via gradient descent.
        //    gradient_w(i->j) = delta_j * activation_i (from the previous layer)
        for (size_t l = 1; l < layers.size(); ++l) {
            Layer& prev = layers[l - 1];
            Layer& curr = layers[l];

            for (Neuron& n : curr.neurons) {
                for (size_t i = 0; i < n.incomingWeights.size(); ++i) {
                    double grad = n.delta * prev.neurons[i].activation;
                    n.weightGradients[i] = grad;
                    n.incomingWeights[i] -= learningRate * grad;
                }
                n.biasGradient = n.delta;
                n.bias -= learningRate * n.delta;
            }
        }

        return loss;
    }

    // Re-randomize all weights/biases (Phase 7: reset button)
    void reset(unsigned seed) {
        rng.seed(seed);
        for (size_t l = 1; l < layers.size(); ++l) {
            int prevSize = layers[l - 1].size();
            for (Neuron& n : layers[l].neurons) {
                n.initWeights(prevSize, rng);
                n.activation = 0.0;
                n.weightedSum = 0.0;
                n.delta = 0.0;
            }
        }
        for (Neuron& n : layers.front().neurons) {
            n.activation = 0.0;
        }
    }

private:
    std::vector<Layer> layers;
    std::mt19937 rng;
};
