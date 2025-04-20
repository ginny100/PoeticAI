# PoeticAI - AI Poetry Generator for Vietnamese

Inspired by [AI for Fun: Làm thơ cùng AI](https://tiensu.github.io/blog/84_make_poem_with_ai/)

This project offers Vietnamese poetry generation using multiple AI approaches, including LSTM and Transformer architectures.

## Setup and Installation

### Requirements

- **Python**: 3.6 - 3.9 (TensorFlow has not officially supported newer versions)
- **Dependencies**: TensorFlow 2.x, NumPy, Matplotlib

### Getting Started

1. First, set up a [Python environment in VS Code](https://code.visualstudio.com/docs/python/environments)
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Helpful Commands

- Check Python version
  ```bash
  python3 --version
  ```

- Check TensorFlow version
  ```bash
  pip3 show tensorflow
  ```

## Poem Generation Models

This project includes two different approaches for Vietnamese poem generation:

1. **LSTM-based Model** (Traditional approach)
2. **Transformer-based Model** (Advanced approach with attention mechanism)

### LSTM-based Generation

The original model using LSTM for sequential poem generation. Best for simpler, shorter poem generation.

To train and use the LSTM model, refer to the basic scripts:
```bash
python train.py --dataset dataset/truyenkieu.txt --epochs 50
python inference.py --input "hạ về xanh biếc trên sông"
```

## Transformer-based Poem Generation

The Transformer architecture leverages attention mechanisms to create high-quality, context-aware poetry in Vietnamese.

### Features

- Transformer architecture for sequence-to-sequence poem generation
- Attention mechanism for improved context understanding
- Support for both training new models and inference with pre-trained models
- Beam search and greedy decoding options for generation
- Temperature control for creativity adjustment
- Visualization of attention weights

### Training a New Model

To train a new Transformer model for poem generation:

```bash
python train_transformer.py --dataset dataset/tonghop.txt --epochs 50 --batch_size 64
```

Options:
- `--dataset`: Path to the dataset file
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--model_dir`: Directory to save the model (default: 'saved_model/poem_transformer')
- `--d_model`: Dimension of model (default: 128)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 4)
- `--dff`: Dimension of feed forward network (default: 512)
- `--dropout_rate`: Dropout rate (default: 0.1)

### Generating Poems

To generate poems using a pre-trained Transformer model:

```bash
python inference_transformer.py --input "hạ về xanh biếc trên sông" --beam_width 5
```

Options:
- `--input`: Input text to start the poem generation
- `--dataset`: Path to the dataset file for tokenizer creation
- `--model_dir`: Directory with pre-trained model (default: 'saved_model/poem_transformer')
- `--beam_width`: Beam width for beam search (set to 1 for greedy search)
- `--temperature`: Temperature for sampling in greedy search (higher = more random)
- `--max_length`: Maximum length of generated poem
- `--max_words`: Maximum number of words in the generated poem
- `--direct`: Use direct generation method (similar to training process)

## Difference between LSTM and Transformer

The Transformer-based poem generation offers several advantages over the LSTM approach:

1. **Parallel processing**: Transformers process the entire sequence in parallel, leading to faster training.
2. **Better long-range dependencies**: The attention mechanism captures relationships between words regardless of their distance in the sequence.
3. **More contextual awareness**: Multi-head attention allows the model to focus on different aspects of the input simultaneously.
4. **Improved quality**: The combination of these features typically results in higher quality poem generation, especially for longer passages.

## Examples

Input:
```
hạ về xanh biếc trên sông
```

Output with Transformer (beam search, width=5):
```
nước chảy mây trôi lặng lẽ trôi
hoa rơi cánh mỏng lá vàng rơi
tình xưa nghĩa cũ còn ghi nhớ
```

Output with LSTM (greedy search):
```
nước chảy mây trôi lặng lẽ trôi
hoa tàn cỏ úa dưới chân đồi
```

## Transformer Architecture

The Transformer architecture follows the original "Attention is All You Need" paper, with:
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Positional encoding

## Acknowledgments

- The Transformer architecture is based on the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- Dataset sources include Vietnamese literature collections