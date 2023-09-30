#ifndef MODEL_H_
#define MODEL_H_

#include <torch/torch.h>

/**
 * @class NeRFModel
 * @brief Represents the Neural Radiance Fields (NeRF) model.
 * 
 * This class extends the torch::nn::Module to utilize functionalities provided by PyTorch.
 */
class NeRFModel : public torch::nn::Module {
public:
  /**
   * @brief Constructor for the NeRFModel class.
   * 
   * @param device The device on which the model will run (CPU or CUDA). Default is CPU.
   * @param positionalLayers The number of positional encoding layers. Default is 6.
   * @param networkDepth The number of layers in the feed-forward network. Default is 8.
   * @param layerWidth The number of neurons in each layer. Default is 256.
   */
  NeRFModel(const torch::Device &device = torch::kCPU, int positionalLayers = 6, int networkDepth = 8, int layerWidth = 256);

  /**
   * @brief Forward pass of the NeRF model.
   * 
   * @param inputTensor The input tensor.
   * @return The output tensor after passing through the model.
   */
  torch::Tensor forward(const torch::Tensor &inputTensor);

  /**
   * @brief Adds positional encoding to the input tensor.
   * 
   * @param inputTensor The input tensor.
   * @return The tensor with positional encoding.
   */
  torch::Tensor addPositionalEncoding(const torch::Tensor &inputTensor) const;

private:
  int positionalLayers_;              ///< Number of positional encoding layers.
  torch::nn::Sequential neuralNet_;   ///< The feed-forward network model.
  const torch::Device &device_;       ///< The device on which the model runs.
};

#endif // MODEL_H_