#ifndef RENDERER_H_
#define RENDERER_H_

#include <torch/torch.h>
#include "model.h"

/**
 * @class NeRFRenderer
 * @brief Responsible for rendering scenes using the Neural Radiance Fields model.
 */
class NeRFRenderer {
public:
  /**
   * @brief Constructor for the NeRFRenderer class.
   * 
   * @param model Reference to the NeRFModel object.
   * @param image_height Height of the rendered image.
   * @param image_width Width of the rendered image.
   * @param focal_length Focal length for the renderer.
   * @param computation_device The device on which the renderer will run (CPU or CUDA).
   */
  NeRFRenderer(NeRFModel &model, int image_height, int image_width, float focal_length, const torch::Device &computation_device);

  /**
   * @brief Renders the scene based on the given pose.
   * 
   * @param pose The pose tensor.
   * @param randomize Whether to randomize the rendering. Default is false.
   * @param start_distance Starting distance for rendering. Default is 2.0f.
   * @param end_distance Ending distance for rendering. Default is 5.0f.
   * @param num_samples Number of samples for rendering. Default is 64.
   * @param batch_size Size of each batch for rendering. Default is 64000.
   * @return The rendered tensor.
   */
  torch::Tensor render(const torch::Tensor &pose, bool randomize = false, float start_distance = 2.0f, float end_distance = 5.0f, int num_samples = 64, int batch_size = 64000) const;

private:
  /**
   * @typedef RayData
   * @brief Type definition for ray data, consisting of ray origins and directions.
   */
  typedef std::tuple<torch::Tensor, torch::Tensor> RayData;

  NeRFModel &model_reference_;          ///< Reference to the NeRFModel object.
  const torch::Device &computation_device_; ///< Device for computation (CPU or CUDA).
  int image_height_;                   ///< Height of the rendered image.
  int image_width_;                    ///< Width of the rendered image.
  float focal_length_;                 ///< Focal length for the renderer.

  /**
   * @brief Computes the rays based on the given pose.
   * 
   * @param pose The pose tensor.
   * @return The computed RayData.
   */
  RayData get_rays(const torch::Tensor &pose) const;

  /**
   * @brief Renders the scene based on the given rays.
   * 
   * @param rays The ray data.
   * @param randomize Whether to randomize the rendering.
   * @param start_distance Starting distance for rendering.
   * @param end_distance Ending distance for rendering.
   * @param num_samples Number of samples for rendering.
   * @param batch_size Size of each batch for rendering.
   * @return The rendered tensor.
   */
  torch::Tensor render_rays(const RayData &rays, bool randomize, float start_distance, float end_distance, int num_samples, int batch_size) const;
};

#endif // RENDERER_H_