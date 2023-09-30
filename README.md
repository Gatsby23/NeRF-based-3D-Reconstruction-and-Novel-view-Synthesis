# NeRF based 3D Reconstruction and Novel view Synthesis

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/ayushgoel24/NeRF-based-3D-Reconstruction-and-Novel-view-Synthesis.svg)](https://github.com/ayushgoel24/NeRF-based-3D-Reconstruction-and-Novel-view-Synthesis/issues)
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-orange.svg)](https://github.com/ayushgoel24/NeRF-based-3D-Reconstruction-and-Novel-view-Synthesis)


This project provides an implementation of a Neural Radiance Field (NeRF) renderer using PyTorch. NeRF is a method for synthesizing novel views of complex scenes by directly optimizing a neural network to represent the scene's volumetric scene function.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
Neural Radiance Fields (NeRF) has emerged as a groundbreaking technique in the realm of 3D scene representation and rendering. By leveraging the power of neural networks, NeRF synthesizes novel views of intricate scenes without the need for traditional 3D modeling. This project offers a streamlined implementation of NeRF, aiming to provide users with the core functionalities of this technique without the overhead of more extensive frameworks. Written in C++, the codebase harnesses the capabilities of LibTorch, PyTorch's C++ API, ensuring efficient computations and leveraging automatic differentiation. Whether you're a researcher looking to experiment with NeRF or a developer aiming to integrate it into a larger system, this project serves as an ideal starting point.

The main functionality of this project is to train a NeRF model using a set of images, poses, and focal lengths. Once trained, the model can be used to render high-resolution images from different viewpoints.

**Key features include:**
- Parsing command-line arguments for data and output paths.
- Setting a random seed for reproducibility.
- Loading data tensors from binary files.
- Training the NeRF model using the Adam optimizer.
- Rendering images using the trained model.
- Saving rendered images to disk.

## Dependencies

To run this project, you'll need the following dependencies:

- **LibTorch**: Version 2.0.0 or higher. LibTorch is the C++ frontend of PyTorch, providing a series of tools and libraries tailored for C++ development. You can download and install it from the official PyTorch website.
- **CUDA** (Optional): If you intend to run the project on an NVIDIA GPU, you'll need CUDA version 11.7 or higher. CUDA allows for parallel computing on NVIDIA GPUs, significantly speeding up computations necessary for large-scale neural networks. Ensure that your GPU is CUDA-compatible and download the necessary toolkit from the [official NVIDIA website](https://developer.nvidia.com/cuda-toolkit).

Please ensure that all dependencies are correctly installed and configured in your environment before running the project.

## Installation

1. **Download and Install LibTorch**:
   - Ensure you have [LibTorch](https://pytorch.org/cppdocs/installing.html) version 2.0.0 or higher.
   - If you place LibTorch in the project root directory, no additional configuration is required.
   - Alternatively, if you choose to install LibTorch locally, remember to update the `CMakeLists.txt` file with the appropriate path to your LibTorch installation.

2. **Clone Repository**: Clone or download the repository into your local machine by running 
    ```bash
    git clone https://github.com/ayushgoel24/NeRF-based-3D-Reconstruction-and-Novel-view-Synthesis.git
    ```

3. **Build the Project Using CMake**:
   Follow the steps below to build the project:
   ```bash
   mkdir -p build && cd build
   cmake ..
   make
    ```

## Usage

After successfully building the project, you can run the executable using the following command:

```bash
./cNeRF /path/to/data /path/to/output
```

Where:

- `/path/to/data` is the path to the directory containing the `images.pt`, `poses.pt`, and `focal.pt` files.
- `/path/to/output` is the path to the directory where rendered images will be saved.

## Results

After running the NeRF renderer, you will obtain synthesized views of your input scenes. These views are generated based on the trained NeRF model and will be saved in the specified output directory. The quality and accuracy of the results depend on the training data and the number of iterations the model has been trained for.

You can visualize the results using any standard image viewer or incorporate them into further post-processing workflows. The generated images offer a novel perspective of the scene, showcasing the power of Neural Radiance Fields in 3D scene representation and rendering.

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any issues, queries, or feedback, please contact: [ayush.goel2427@gmail.com](mailto:ayush.goel2427@gmail.com).