#include "utils.h"

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace Utils {

    /** 
     * @brief Set up the random seed for reproducibility.
     * 
     * @param seed The seed value.
     */
    void setSeed(int seed) {
        torch::manual_seed(seed);
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed(seed);
        }
    }

    /** 
     * @brief Determine the appropriate device for computation (CPU or GPU).
     * 
     * @return The computation device.
     */
    torch::Device getDevice() {
        return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    }

    /** 
     * @brief Parse command-line arguments.
     * 
     * @param argc Argument count.
     * @param argv Argument values.
     * @param dataPath Path to the data directory.
     * @param outputPath Path to the output directory.
     * @return True if arguments are parsed successfully, false otherwise.
     */
    bool parseArguments(int argc, char *argv[], std::filesystem::path &dataPath, std::filesystem::path &outputPath) {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <dataPath> <outputPath>" << std::endl;
            return false;
        }

        dataPath = argv[1];
        outputPath = argv[2];
        return true;
    }

    /** 
     * @brief Load a binary file.
     * 
     * @param filePath Path to the binary file.
     * @return Vector containing the file's binary data.
     */
    std::vector<char> loadBinaryFile(const std::filesystem::path &filePath) {
        std::ifstream input(filePath, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                (std::istreambuf_iterator<char>()));
        input.close();
        return bytes;
    }

    /** 
     * @brief Load a tensor from a file.
     * 
     * @param filePath Path to the tensor file.
     * @return The loaded tensor.
     */
    torch::Tensor loadTensor(const std::filesystem::path &filePath) {
        std::vector<char> fileData = loadBinaryFile(filePath);
        torch::IValue tensorValue = torch::pickle_load(fileData);
        return tensorValue.toTensor();
    }

    /** 
     * @brief Load focal length from a file.
     * 
     * @param filePath Path to the focal length file.
     * @return The loaded focal length.
     */
    float loadFocal(const std::filesystem::path &filePath) {
        torch::Tensor focalTensor = loadTensor(filePath);
        return focalTensor.item<float>();
    }

    /** 
     * @brief Save an image tensor to a file.
     * 
     * @param imageTensor The image tensor.
     * @param filePath Path to save the image.
     */
    void saveImage(const torch::Tensor &imageTensor, const std::filesystem::path &filePath) {
        // Assuming the input tensor is a 3-channel (HxWx3) image in the range [0, 1]
        auto height = imageTensor.size(0);
        auto width = imageTensor.size(1);
        auto maxVal = imageTensor.max();
        auto minVal = imageTensor.min();
        auto tensorNormalized = ((imageTensor - minVal) / (maxVal - minVal))
                                    .mul(255)
                                    .clamp(0, 255)
                                    .to(torch::kU8)
                                    .to(torch::kCPU)
                                    .flatten()
                                    .contiguous();
        cv::Mat image(cv::Size(width, height), CV_8UC3, tensorNormalized.data_ptr());
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imwrite(filePath.string(), image);
    }

    /** 
     * @brief Render and save orbit views of a scene.
     * 
     * @param renderer The NeRF renderer.
     * @param numFrames The number of frames to render.
     * @param outputFolder The folder to save the rendered images.
     * @param radius The radius of the orbit. Default is 4.0f.
     * @param startDistance The starting distance for rendering. Default is 2.0f.
     * @param endDistance The ending distance for rendering. Default is 5.0f.
     * @param numSamples The number of samples for rendering. Default is 64.
     */
    void renderAndSaveOrbitViews(const NeRFRenderer &renderer, int numFrames,
                                 const std::filesystem::path &outputFolder,
                                 float radius = 4.0f, float startDistance = 2.0f,
                                 float endDistance = 5.0f, int numSamples = 64) {
        float elevation = -30.0f;

        for (int i = 0; i < numFrames; i++) {
            float azimuth = static_cast<float>(i) * 360.0f / numFrames;
            auto pose = createSphericalPose(azimuth, elevation, radius);

            auto renderedImage =
                renderer.render(pose, false, startDistance, endDistance, numSamples);

            std::string filePath =
                outputFolder / ("frame_" + std::to_string(i) + ".png");
            saveImage(renderedImage, filePath);
        }
    }

    /** 
     * @brief Create a spherical pose matrix.
     * 
     * @param azimuth The azimuth angle in degrees.
     * @param elevation The elevation angle in degrees.
     * @param radius The radius of the sphere.
     * @return The spherical pose matrix.
     */
    torch::Tensor createSphericalPose(float azimuth, float elevation, float radius) {
        float phi = elevation * (M_PI / 180.0f);
        float theta = azimuth * (M_PI / 180.0f);

        torch::Tensor c2w = createTranslationMatrix(radius);
        c2w = createPhiRotationMatrix(phi).matmul(c2w);
        c2w = createThetaRotationMatrix(theta).matmul(c2w);
        c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                             {0.0f, 0.0f, 1.0f, 0.0f},
                             {0.0f, 1.0f, 0.0f, 0.0f},
                             {0.0f, 0.0f, 0.0f, 1.0f}})
                  .matmul(c2w);

        return c2w;
    }

    /** 
     * @brief Create a translation matrix.
     * 
     * @param translationValue The value by which to translate.
     * @return The translation matrix.
     */
    torch::Tensor createTranslationMatrix(float translationValue) {
        torch::Tensor translationMat = torch::tensor({
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, translationValue},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        return translationMat;
    }

    /** 
     * @brief Create a phi rotation matrix.
     * 
     * @param phi The phi angle in radians.
     * @return The phi rotation matrix.
     */
    torch::Tensor createPhiRotationMatrix(float phi) {
        torch::Tensor phiMat = torch::tensor({
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
            {0.0f, std::sin(phi), std::cos(phi), 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        return phiMat;
    }

    /** 
     * @brief Create a theta rotation matrix.
     * 
     * @param theta The theta angle in radians.
     * @return The theta rotation matrix.
     */
    torch::Tensor createThetaRotationMatrix(float theta) {
        torch::Tensor thetaMat = torch::tensor({
            {std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        return thetaMat;
    }

} // namespace Utils