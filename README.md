# ast_inst_cls

# Reprogramming a Pre-trained Model for Music Classification Tasks

## Problems of music classification task
Challenge of collecting large-scale datasets
- Resource consuming
- Experts needed -> timbre, genre, etc.
- Crowd sourcing -> Annotation might not be robust

Multi-label problem
 - Polyphonic audios
 - Instrument identification, music tagging -> not fully labeled
 - Missing labels can mean either present or absent instances  -> problematic!

How to detect and classify with insufficient labels?​
- Complex model?
- Advanced semi/self-supervised learning (SSL) approach to handle missing labels
- What else can we do?


## Reprogramming
Leveraging the power of pre-trained models
 - Well-studied domains​
   - Image, speech, …
 - 'Reprogram' the pre-trained model (black-box)

<img src="https://github.com/hchen605/ast_inst_cls/blob/master/fig/reprogramming%20blk.png" width="9500" height="400" />

## Preliminary results on music instrument classification

Polyphonic music instrument classification​

 - OpenMIC Dataset (20000 data)​

 - Weakly labeled: 90% of the labels are missing​

Pre-trained model

- Audio Spectrogram Transformers (AST)​

- SOTA evaluation over AudioSet​

Baseline​

- Input + pre-trained model + linear layer​

Reprogramming method​

1. Adversarial programming​

2. Noise addition​

3. Input CNN transformation​

w/ simple output linear layer label mapping

<img src="https://github.com/hchen605/ast_inst_cls/blob/master/fig/reprogramming%20result.png" width="400" height="400" />
