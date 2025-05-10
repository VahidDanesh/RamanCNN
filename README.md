# Ex-vivo Raman Spectroscopy and AI-based Classification of Soft Tissue Sarcomas 

This repository contains code for analyzing Raman spectroscopy data and classifying different tissue types using deep learning. The project implements a ResNet-based model for distinguishing between various tissue types including Normal, Benign, and Malignant tissues.

## Getting Started

### Clone the Repository

```bash
# Clone the repository
git clone https://github.com/VahidDanesh/RamanCNN.git
cd RamanCNN
```

## Installation

You can set up the required dependencies using either conda or uv.

### Using conda

```bash
# Create a new conda environment
conda create -n raman-cnn python=3.10
conda activate raman-cnn

# Install PyTorch with CUDA support (if applicable)
conda install pytorch torchvision torchaudio
# Install other dependencies
conda install pandas numpy matplotlib seaborn scikit-learn
```

### Using uv

It's highly recommended to use `uv`, a fast and efficient Python package installer.
```bash
# Install uv if you don't have it already
pip install uv

# Create and activate a new virtual environment
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Quick install all dependencies from pyproject.toml
uv pip install -e .

# Or install dependencies individually
uv pip install torch torchvision torchaudio
uv pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Extract the dataset in the `/data/dataset/` directory
2. Run the main notebook to train the model:
   ```
   main.ipynb
   ```
3. Evaluate the model performance:
   ```
   eval.ipynb
   ```

## Project Structure

- `main.ipynb`: Contains code for data preparation and model training
- `eval.ipynb`: Evaluates the trained model and generates visualizations
- `models/`: Contains model architectures
- `utils/`: Helper functions for data processing
- `config.py`: Configuration parameters for the project

## Citation

If you find this code helpful, please consider citing our paper:

```
Boroji, M., Danesh, V., Barrera, D., Lee, E., Arauz, P., Farrell, R., Boyce, B., Khan, F., and Kao, I. "Ex-Vivo Raman Spectroscopy and AI-Based Classification of Soft Tissue Sarcomas", PLOS ONE, Public Library of Science, 2025
```