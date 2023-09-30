#ifndef UTILS_H_
#define UTILS_H_

#include <filesystem>
#include <torch/torch.h>
#include "renderer.h"

/**
 * @namespace Utils
 * @brief Utility functions for NeRF project.
 */
namespace Utils {

    /**
     * @brief Set the random seed for reproducibility.
     * 
     * @param seed The seed value.
     */
    void setSeed(int seed);

    /**
     * @brief Determine the appropriate device for computation (CPU or GPU).
     * 
     * @return The computation device.
     */
    torch::Device getDevice();

    /**
     * @brief Parse command-line arguments.
     * 
     * @param argc Argument count.
     * @param argv Argument values.
     * @param dataPath Path to the data directory.
     * @param outputPath Path to the output directory.
     * @return True if arguments are parsed successfully, false otherwise.
     */
    bool parseArguments(int argc, char *argv[], std::filesystem::path &dataPath, std::filesystem::path &outputPath);

    /**
     * @brief Load a binary file.
     * 
     * @param filePath Path to the binary file.
     * @return Vector containing the file's binary data.
     */
    std::vector<char> loadBinaryFile(const std::filesystem::path &filePath);

    /**
     * @brief Load a tensor from a file.
     * 
     * @param filePath Path to the tensor file.
     * @return The loaded tensor.
     */
    torch::Tensor loadTensor(const std::filesystem::path &filePath);

    /**
     * @brief Load focal length from a file.
     * 
     * @param filePath Path to the focal length file.
     * @return The loaded focal length.
     */
    float loadFocal(const std::filesystem::path &filePath);

    /**
     * @brief Save an image tensor to a file.
     * 
     * @param tensor The image tensor.
     * @param filePath Path to save the image.
     */
    void saveImage(const torch::Tensor &tensor, const std::filesystem::path &filePath);

    /**
     * @brief Render and save orbit views.
     * 
     * @param renderer The NeRF renderer.
     * @param numFrames Number of frames to render.
     * @param outputFolder Output directory to save the rendered images.
     * @param radius Radius for orbiting. Default is 4.0f.
     * @param startDistance Start distance for rendering. Default is 2.0f.
     * @param endDistance End distance for rendering. Default is 5.0f.
     * @param numSamples Number of samples for rendering. Default is 64.
     */
    void renderAndSaveOrbitViews(const NeRFRenderer &renderer, int numFrames, const std::filesystem::path &outputFolder, float radius = 4.0f, float startDistance = 2.0f, float endDistance = 5.0f, int numSamples = 64);

    /**
     * @brief Create a spherical pose tensor.
     * 
     * @param azimuth Azimuth angle.
     * @param elevation Elevation angle.
     * @param radius Radius value.
     * @return The spherical pose tensor.
     */
    torch::Tensor createSphericalPose(float azimuth, float elevation, float radius);

    /**
     * @brief Create a translation matrix tensor.
     * 
     * @param t Translation value.
     * @return The translation matrix tensor.
     */
    torch::Tensor createTranslationMatrix(float t);

    /**
     * @brief Create a phi rotation matrix tensor.
     * 
     * @param phi Phi angle.
     * @return The phi rotation matrix tensor.
     */
    torch::Tensor createPhiRotationMatrix(float phi);

    /**
     * @brief Create a theta rotation matrix tensor.
     * 
     * @param theta Theta angle.
     * @return The theta rotation matrix tensor.
     */
    torch::Tensor createThetaRotationMatrix(float theta);

} // namespace Utils

#endif // UTILS_H_