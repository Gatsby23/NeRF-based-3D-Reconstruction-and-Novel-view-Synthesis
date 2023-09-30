#include "model.h"
#include "renderer.h"
#include "utils.h"

// Constants
constexpr int SEED = 1;
constexpr int NUM_ITERATIONS = 2000;
constexpr int LOG_FREQUENCY = 100;
constexpr int PREVIEW_FRAME_COUNT = 5;
constexpr int FINAL_FRAME_COUNT = 35;

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    std::filesystem::path dataPath;
    std::filesystem::path outputPath;
    if (!parseArguments(argc, argv, dataPath, outputPath)) {
        return 1;
    }

    // Set the random seed for reproducibility
    setSeed(SEED);

    // Determine the computation device (CPU or GPU)
    torch::Device device = getDevice();

    // Load data: images, camera poses, and focal length
    torch::Tensor images = loadTensor(dataPath / "images.pt").to(device);
    torch::Tensor poses = loadTensor(dataPath / "poses.pt").to(device);
    float focalLength = loadFocal(dataPath / "focal.pt");

    // Display information about the loaded data
    std::cout << "Images: " << images.sizes() << std::endl;
    std::cout << "Poses: " << poses.sizes() << std::endl;
    std::cout << "Focal Length: " << focalLength << std::endl;

    // Initialize NeRF model and renderer
    NeRFModel model(device);
    NeRFRenderer renderer(model, images.size(1), images.size(2), focalLength, device);

    // Set up the optimizer for training
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(5e-4));

    // Train the NeRF model
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Randomly sample an image and its corresponding camera pose
        int randomImageIndex = std::rand() % images.size(0);
        auto targetImage = images[randomImageIndex];
        auto cameraPose = poses[randomImageIndex];

        // Forward pass through the model and compute the loss
        optimizer.zero_grad();
        auto renderedRGB = renderer.render(cameraPose, true);
        auto loss = torch::mse_loss(renderedRGB, targetImage);

        // Backward pass and update model parameters
        loss.backward();
        optimizer.step();

        // Log progress at specified intervals
        if (i % LOG_FREQUENCY == 0) {
            torch::NoGradGuard noGrad;
            std::cout << "Iteration: " << i + 1 << " Loss: " << loss.item<float>() << std::endl;

            // Render and save preview images for visualization
            renderAndSaveOrbitViews(renderer, PREVIEW_FRAME_COUNT, outputPath, 4.0f);
        }
    }

    std::cout << "Training Completed!" << std::endl;

    // Generate high-resolution renderings using the trained model
    torch::NoGradGuard noGrad;
    NeRFRenderer highResRenderer(model, 300, 300, focalLength, device);
    renderAndSaveOrbitViews(highResRenderer, FINAL_FRAME_COUNT, outputPath, 2.1f, 0.8f, 3.2f);

    return 0;
}