#include "renderer.h"
#include "utils.h"

using namespace torch::indexing;

namespace Renderer {

    NeRFRenderer::NeRFRenderer(NeRFModel &model, int imageHeight, int imageWidth, float focalLength,
                               const torch::Device &device)
        : model_(model), imageHeight_(imageHeight), imageWidth_(imageWidth), focalLength_(focalLength), device_(device) {}

    /** 
     * @brief Render an image using the NeRF model.
     * 
     * @param pose The pose matrix.
     * @param randomize Whether to randomize the z-values.
     * @param startDistance The starting distance for ray marching.
     * @param endDistance The ending distance for ray marching.
     * @param numSamples The number of samples for ray marching.
     * @param batchSize The batch size for processing points.
     * @return The rendered image.
     */
    torch::Tensor NeRFRenderer::render(const torch::Tensor &pose, bool randomize,
                                       float startDistance, float endDistance,
                                       int numSamples, int batchSize) const {
        auto rays = getRays(pose.to(device_));
        return renderRays(rays, randomize, startDistance, endDistance, numSamples, batchSize);
    }

    /** 
     * @brief Get rays for rendering.
     * 
     * @param pose The pose matrix.
     * @return The ray data.
     */
    NeRFRenderer::RayData NeRFRenderer::getRays(const torch::Tensor &pose) const {
        // Generate pixel indices along image width (i) and height (j)
        auto i = torch::arange(imageWidth_, torch::dtype(torch::kFloat32)).to(device_);
        auto j = torch::arange(imageHeight_, torch::dtype(torch::kFloat32)).to(device_);
        auto grid = torch::meshgrid({i, j}, "xy");
        auto ii = grid[0];
        auto jj = grid[1];

        // Compute the direction vector for each pixel in the image plane
        auto dirs = torch::stack({(ii - imageWidth_ * 0.5) / focalLength_, 
                                  -(jj - imageHeight_ * 0.5) / focalLength_, 
                                  -torch::ones_like(ii)},
                                 -1);

        // Transform the direction vectors from the camera's local coordinate system
        // to the global coordinate system
        auto raysDirection = torch::sum(dirs.index({"...", None, Slice()}) *
                                        pose.index({Slice(0, 3), Slice(0, 3)}),
                                        -1);
        // Get the origin of the rays from the pose
        auto raysOrigin = pose.index({Slice(0, 3), -1}).expand(raysDirection.sizes());

        return std::make_tuple(raysOrigin, raysDirection);
    }

    /** 
     * @brief Render rays to produce an image.
     * 
     * @param rays The ray data.
     * @param randomize Whether to randomize the z-values.
     * @param startDistance The starting distance for ray marching.
     * @param endDistance The ending distance for ray marching.
     * @param numSamples The number of samples for ray marching.
     * @param batchSize The batch size for processing points.
     * @return The rendered image.
     */
    torch::Tensor NeRFRenderer::renderRays(const RayData &rays, bool randomize,
                                           float startDistance, float endDistance, 
                                           int numSamples, int batchSize) const {
        // Unpack the ray origins and directions
        auto raysOrigin = std::get<0>(rays);
        auto raysDirection = std::get<1>(rays);

        // Compute 3D query points
        auto zVals = torch::linspace(startDistance, endDistance, numSamples, device_)
                        .reshape({1, 1, numSamples})
                        .expand({imageHeight_, imageWidth_, numSamples})
                        .clone();
        if (randomize) {
            zVals += torch::rand({imageHeight_, imageWidth_, numSamples}, device_) *
                     (startDistance - endDistance) / numSamples;
        }
        auto pts = raysOrigin.unsqueeze(-2) + raysDirection.unsqueeze(-2) * zVals.unsqueeze(-1);

        // Encode points
        auto ptsFlat = pts.view({-1, 3});
        auto ptsEmbedded = model_.add_positional_encoding(ptsFlat);

        // Batch-process points
        int nPts = ptsFlat.size(0);
        torch::Tensor raw;
        for (int i = 0; i < nPts; i += batchSize) {
            auto batch = ptsEmbedded.slice(0, i, std::min(i + batchSize, nPts));
            auto batchRaw = model_.forward(batch);
            if (i == 0) {
                raw = batchRaw;
            } else {
                raw = torch::cat({raw, batchRaw}, 0);
            }
        }
        raw = raw.view({imageHeight_, imageWidth_, numSamples, 4});

        // Get volume colors and opacities
        auto rgb = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
        auto sigmaA = torch::relu(raw.index({"...", 3}));

        // Render volume
        auto dists = torch::cat({zVals.index({"...", Slice(1, None)}) -
                                 zVals.index({"...", Slice(None, -1)}),
                                 torch::full({1}, 1e10, device_).expand({imageHeight_, imageWidth_, 1})},
                                -1);
        auto alpha = 1.0 - torch::exp(-sigmaA * dists);
        auto weights = torch::cumprod(1.0 - alpha + 1e-10, -1);
        weights = alpha * torch::cat({torch::ones({imageHeight_, imageWidth_, 1}, device_),
                                      weights.index({"...", Slice(None, -1)})},
                                     -1);

        auto rgbMap = torch::sum(weights.unsqueeze(-1) * rgb, -2);
        auto depthMap = torch::sum(weights * zVals, -1);
        auto accMap = torch::sum(weights, -1);

        return rgbMap;
    }

} // namespace Renderer