#include "model.h"

/**
 * @brief Construct a new NeRF Model object.
 * 
 * @param device The device to run the model on (e.g., CPU or CUDA).
 * @param positionalEmbeddingLayers Number of positional embedding layers.
 * @param depth Depth of the feed-forward network.
 * @param width Width of the feed-forward network.
 */
NeRFModel::NeRFModel(const torch::Device &device, int positionalEmbeddingLayers, int depth, int width)
    : device_(device), positionalEmbeddingLayers_(positionalEmbeddingLayers) {
    
    // Create feed-forward network (FFN)
    int inputDimension = 3 + 3 * 2 * positionalEmbeddingLayers;
    model_->push_back(torch::nn::Linear(inputDimension, width));
    model_->push_back(torch::nn::Functional(torch::relu));
    
    for (int i = 0; i < depth - 2; i++) {
        model_->push_back(torch::nn::Linear(width, width));
        model_->push_back(torch::nn::Functional(torch::relu));
    }
    
    model_->push_back(torch::nn::Linear(width, 4));
    model_->to(device_);
    register_module("model", model_);
    this->to(device_);
}

/**
 * @brief Forward pass of the NeRF model.
 * 
 * @param input The input tensor.
 * @return The output tensor.
 */
torch::Tensor NeRFModel::forward(const torch::Tensor &input) {
    return model_->forward(input);
}

/**
 * @brief Add positional encoding to the input tensor.
 * 
 * @param x The input tensor.
 * @return The tensor with positional encoding.
 */
torch::Tensor NeRFModel::addPositionalEncoding(const torch::Tensor &x) const {
    std::vector<torch::Tensor> encodings = {x};
    
    for (int i = 0; i < positionalEmbeddingLayers_; i++) {
        encodings.push_back(torch::sin(std::pow(2.0f, i) * x));
        encodings.push_back(torch::cos(std::pow(2.0f, i) * x));
    }
    
    return torch::cat(encodings, -1);
}