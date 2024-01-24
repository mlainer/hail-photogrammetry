This Docker image is based on nvcr.io/nvidia/cuda:11.2.2-base-ubuntu20.04 and is optimized for machine learning and computer vision tasks. It includes essential dependencies and tools for developing and running applications using Nvidia CUDA. The image is configured with Python 3, Jupyter Lab, and popular machine learning libraries such as PyTorch and Detectron2.

Features: Base Image: Built on the official Nvidia CUDA image, providing a robust foundation for GPU-accelerated workloads.

Development Environment: Python 3, pip, and essential development tools are pre-installed, making it easy to set up and work on machine learning projects.

Machine Learning Libraries: Includes popular Python libraries such as NumPy, Matplotlib, scikit-learn, scikit-image, OpenCV (with and without contrib modules), and Jupyter Lab for a comprehensive development environment.

PyTorch with CUDA 11.1: PyTorch and related packages are installed with GPU support, utilizing CUDA 11.1 for optimized performance.

Detectron2 Framework: The Detectron2 framework from Facebook Research is included and installed as an editable package, allowing users to easily experiment with state-of-the-art object detection models.

Port Exposed: Port 8822 is exposed for Jupyter Lab, making it accessible for interactive development and experimentation.
