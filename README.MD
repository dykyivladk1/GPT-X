# GPT-X

This repository contains the implementation of GPT-based model for text generation, including the training inference pipeline. The project is build using PyTorch and includes custom dataset preparation, model architecture, and utilities for training and testing the model.



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)




## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dykyivladk1/GPT-X.git
   cd GPT-X
    ```


2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure that PyTorch is installed. You can install it using the official guide [here](https://pytorch.org/get-started/locally/).


3. Prepare your dataset (text file):
   - Place your text data in a `.txt` file. The default path is `./data/sample.txt`.

## Usage

### Training

To train the model, use the following command:

```bash
python train.py --data_path /path/to/your/dataset.txt --max_iters 10000 --block_size 128 --batch_size 64 --learning_rate 3e-4 --betas 0.9 0.95 --weight_decay 0.1 --grad_norm_clip 1.0 --mode training
```

**Arguments:**

- `--data_path`: Path to the text file containing training data.
- `--max_iters`: Maximum number of training iterations.
- `--block_size`: Length of the text sequence to process.
- `--batch_size`: Number of samples per batch.
- `--learning_rate`: Learning rate for the optimizer.
- `--betas`: Betas for the Adam optimizer.
- `--weight_decay`: Weight decay for regularization.
- `--grad_norm_clip`: Maximum norm for gradient clipping.
- `--mode`: Mode of operation, set to `training` for training.


### Testing

To test the model and generate text, use:

```bash
python train.py --mode test --context "Sample Input" --weights_path weights/model.pth
```

**Arguments:**

- `--mode`: Set to `test` for text generation.
- `--context`: Initial text to start the generation.
- `--weights_path`: Path to the model weights file.
- `--max_tokens`: Maximum number of tokens to generate.

## Project Structure

```
GPT-X/
├── src/                   
│   ├── dataset.py         
│   ├── encoder_.py       
│   ├── main.py            
│   ├── model.py           
│   └── utils.py          
├── weights/         
│   ├── dataset.py     
└── README.md              
```

## Model Overview

The GPT-X model is a simplified version of the GPT architecture, designed to generate text based on a given context. It includes the following components:

- **Multi-head Self-Attention**
- **Positional Encoding**
- **Feedforward Layers**
- **Layer Normalization**

The model is built with flexibility in mind, allowing easy adjustments to the number of layers, heads, and embedding size.

## Dataset

The dataset is a simple text file where the model will learn to predict the next character in a sequence based on previous characters. The dataset is tokenized and prepared using the `GPT_XDataset` class.

**Dataset Class:**

- `GPT_XDataset`: Prepares sequences of text for training the GPT-X model. It converts characters into indices and creates input-target pairs for training.

## Contributing

Contributions are welcome! Please open an issue or a pull request for any improvements or suggestions.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes.
4. Open a pull request.





