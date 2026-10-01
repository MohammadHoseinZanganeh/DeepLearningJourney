# Project04 — RNN & Vision Transformer

This project contains two independent parts:

## Part 1 — Named Entity Recognition (NER)

Implementation and comparison of three bidirectional RNN architectures (**BiRNN**, **BiLSTM**, **BiGRU**) on the **CoNLL-2003** dataset for the Named Entity Recognition task.

- Encoder: Embedding → Bidirectional RNN → Linear
- Labels: `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC`
- Configurable via `config/config.yaml`

📄 Full details: [homework_code/part1/README.md](homework_code/Part1/README.md)

## Part 2 — Vision Transformer (ViT) for CIFAR-10

A from-scratch implementation of the **Vision Transformer** (Dosovitskiy et al., 2021) in PyTorch, trained on **CIFAR-10**, including:

- Patch Embedding, Multi-Head Self-Attention, Transformer Blocks
- Hyperparameter experiments (`embed_dim`, `num_layers`, `num_heads`, `patch_size`)
- Attention map visualization for both the scratch model and pretrained ViT-B/16

📄 Full details: [homework_code/part2/README.md](homework_code/Part2/README.md)

---

## Project Structure
