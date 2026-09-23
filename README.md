# Transformers From Scratch

This repository contains a minimalist, from-scratch implementation of a Transformer neural network. The model is designed to perform character-level language modeling and text generation, specifically trained on the Tiny Shakespeare dataset. It serves as an educational resource for understanding the underlying architecture of modern Large Language Models (LLMs).

## Repository Structure

* **`Tiny_Shakespear_Transformer.ipynb`**[cite: 1]: An interactive Jupyter Notebook that breaks down the Transformer architecture step-by-step. It includes the data loader, self-attention mechanisms, feed-forward networks, and the training loop.
* **`script.py`**[cite: 1]: A standalone, executable Python script containing the consolidated model architecture and training pipeline for terminal-based execution.
* **`input.txt`**[cite: 1]: The training dataset containing the works of Shakespeare concatenated into a single text file. 

## Getting Started

### Prerequisites
To run the notebook or script, you will need an environment with standard deep learning and data science libraries installed. It is recommended to use an environment with:
* Python 3.8+
* PyTorch
* Jupyter Notebook (for the `.ipynb` file)

### Usage

**Option 1: Interactive Exploration**
Launch Jupyter Notebook and open the implementation to run the model cell-by-cell and visualize the training process:
```bash
jupyter notebook Tiny_Shakespear_Transformer.ipynb
