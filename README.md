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

### 1-Using conda

```bash
# Create a new conda environment
conda create -n raman-cnn python=3.10
conda activate raman-cnn

# Install PyTorch with CUDA support (if applicable)
conda install pytorch torchvision torchaudio
# Install other dependencies
conda install pandas numpy matplotlib seaborn scikit-learn
```

### 2-Using uv

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

1. Download the [RamanSTS dataset](https://doi.org/10.7910/DVN/NGEZPA) from Harvard Dataverse into `data/dataset/` (about 2.4 GB download, 3.2 GB extracted):
   ```bash
   python download_data.py
   ```
   The script downloads the two archives, checks their MD5 checksums, extracts them and deletes the archives (use `--keep-zips` to keep them). It only needs the Python standard library.
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
- `download_data.py`: Downloads the RamanSTS dataset from Harvard Dataverse into `data/dataset/`

## Citation

If you find this code helpful, please consider citing our paper:

```bibtex
@article{Boroji2025,
  author    = {Boroji, M. and Danesh, V. and Barrera, D. and Lee, E. and Arauz, P. G. and Farrell, R. F. and others},
  title     = {Ex-vivo Raman spectroscopy and AI-based classification of soft tissue sarcomas},
  journal   = {PLOS ONE},
  year      = {2025},
  volume    = {20},
  number    = {8},
  pages     = {},
  doi       = {10.1371/journal.pone.0330618},
  url       = {https://doi.org/10.1371/journal.pone.0330618}
}
```
