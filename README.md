# PyTorch Transformer: English to Italian Translation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A Transformer-based English-to-Italian machine translation model implemented from scratch with PyTorch. The project follows the encoder-decoder Transformer architecture introduced in *Attention Is All You Need* by Vaswani et al. and is trained on the `Helsinki-NLP/opus_books` dataset.

The implementation uses custom PyTorch modules rather than high-level Transformer implementations.

## Features

* Transformer encoder-decoder architecture
* Six encoder layers and six decoder layers
* Multi-head attention with eight attention heads
* Sinusoidal positional encoding
* Custom layer normalization
* Residual connections and dropout
* Label smoothing
* Word-level tokenization
* Greedy decoding and beam search experimentation
* TensorBoard logging
* Weights & Biases logging
* Streamlit interface for translation
* Training support for CPU, CUDA, and Apple Silicon MPS

## Project Structure

```text
pytorch-transformer-main/
├── app.py
├── config.py
├── model.py
├── dataset.py
├── train.py
├── train_wb.py
├── translate.py
├── requirements.txt
├── weights/
├── opus_books_weights/
├── tokenizer_en.json
├── tokenizer_it.json
├── runs/
├── *.ipynb
│   ├── attention_visual.ipynb
│   ├── Beam_Search.ipynb
│   ├── Colab_Train_fixed.ipynb
│   ├── Inference.ipynb
│   └── Local_Train.ipynb
└── venv/
```

## Model Architecture

The model follows the original Transformer architecture with separate encoder and decoder stacks.

| Component               | Configuration                      |
| ----------------------- | ---------------------------------- |
| Encoder Layers          | 6                                  |
| Decoder Layers          | 6                                  |
| Embedding Dimension     | 512                                |
| Feed-Forward Dimension  | 2048                               |
| Attention Heads         | 8                                  |
| Dropout                 | 0.1                                |
| Label Smoothing         | 0.1                                |
| Maximum Sequence Length | 350                                |
| Optimizer               | Adam                               |
| Learning Rate           | 1e-4                               |
| Adam Epsilon            | 1e-9                               |
| Loss Function           | Cross-Entropy with Label Smoothing |

### Implemented Components

The Transformer architecture is built from custom PyTorch modules:

* `LayerNormalization`
* `MultiHeadAttentionBlock`
* `FeedForwardBlock`
* `PositionalEncoding`
* `ResidualConnection`
* `Encoder`
* `Decoder`

## Training

### Dataset

The model is trained using the `Helsinki-NLP/opus_books` English-Italian dataset.

* Training split: 90%
* Validation split: 10%
* Batch size: 8
* Training epochs: 20–30
* Maximum sequence length: 350
* Optimizer: Adam
* Learning rate: `1e-4`

Training can be performed using CPU, CUDA, or Apple Silicon MPS.

### Logging

Two training scripts are provided:

* `train.py` — standard training with TensorBoard logging
* `train_wb.py` — training with Weights & Biases logging

The project also tracks translation-related validation metrics including BLEU, Character Error Rate (CER), and Word Error Rate (WER).

## Results

This project is primarily an educational implementation of the Transformer architecture.

The trained model learns basic English-to-Italian translation patterns and vocabulary, but translation quality is limited, particularly for longer or more complex sentences. The model's reported BLEU score is below 10.

The implementation demonstrates the complete training and inference pipeline while highlighting the computational and data requirements of Transformer-based machine translation.

## Installation

Clone the repository:

```bash
git clone https://github.com/alibahja/Pytorch-transformer.git
cd Pytorch-transformer-main
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset and Tokenization

Run the dataset preparation script to download the dataset and train the tokenizers:

```bash
python dataset.py
```

## Training

### TensorBoard

```bash
python train.py
```

To monitor training:

```bash
tensorboard --logdir runs/
```

### Weights & Biases

Authenticate with Weights & Biases:

```bash
wandb login
```

Then start training:

```bash
python train_wb.py
```

## Streamlit Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application provides an interface for entering English text and viewing the model's Italian translation.

## Inference

Translation utilities are provided through `translate.py`.

```python
from translate import translate

result = translate("Hello, how are you today?")
print(result)
```

## Notebooks

The repository includes notebooks for experimentation and training:

* `Inference.ipynb` — inference experiments
* `Beam_Search.ipynb` — beam search experiments
* `attention_visual.ipynb` — attention visualization
* `Colab_Train_fixed.ipynb` — training on Google Colab
* `Local_Train.ipynb` — local training experiments

## Configuration

Model and training parameters are defined in `config.py`.

The configuration includes parameters such as:

* `batch_size`
* `num_epochs`
* `lr`
* `seq_len`
* `d_model`
* `d_ff`
* source and target languages
* dataset configuration

## Limitations

The model is trained with limited computational resources and a relatively small dataset compared with large-scale machine translation systems.

Current limitations include:

* Limited performance on long or complex sentences
* Difficulty with contextual and idiomatic expressions
* Word-level tokenization limitations, particularly with rare or unseen words
* Translation quality that is not suitable for production use

## Possible Improvements

Potential directions for further development include:

* Subword tokenization using BPE, WordPiece, or SentencePiece
* Learning rate warmup
* Training on a larger dataset
* Longer training
* Additional attention visualizations
* Experimentation with different optimization and regularization settings
* Alternative Transformer normalization architectures

## References

* Vaswani et al. — *Attention Is All You Need*
* The Annotated Transformer
* `Helsinki-NLP/opus_books`
* PyTorch documentation and tutorials

## License

This project is licensed under the MIT License.
