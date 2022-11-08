# Music Instrument Classification Reprogrammed

#### Reprogramming a Pre-trained Model for Music Classification Tasks

This repo contains the background and implementation (in PyTorch) of the paper: "Music Instrument Classification Reprogrammed" proposed in the 2023 International Conference of MultiMedia Modelling (MMM23).


## Usage

Please install the environment with the `ast_repr.yml`

Command: 
- `cd egs/openmic`
- source `run.sh`

The default will run the U-Net reprogramming evaluation over OpenMic dataset.

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

Adopted/Proposed Reprogramming method​

1. Adversarial programming​

2. Noise reprogramming​

3. Input CNN transformation​

4. Input U-Net transformation​

w/ simple output linear layer label mapping
## Music instrument classification results

Polyphonic music instrument classification​

 - OpenMIC Dataset (20000 data)​

 - Weakly labeled: 90% of the labels are missing​

Pre-trained model

- Audio Spectrogram Transformers (AST)​

- SOTA evaluation over AudioSet/SpeechCommand​

Baseline​

- Random Forest (RF)
- CNN baseline (CNN-BS)
- AST pre-trained model (AST-BS)​
- Mean Teacher (MT)

Proposal
- AST transfer learning (AST-TL)
- AST + Noise Reprogramming (AST-NRP)
- AST + CNN Reprogramming (AST-CNNRP)
- AST + U-Net Reprogramming (AST-URP)

#### The F1-score comparison:
<img src="https://github.com/hchen605/ast_inst_cls/blob/master/fig/Ins_f1_box_rf.png" width="700" height="300" />

#### Instrument-wise F1-score:
<img src="https://github.com/hchen605/ast_inst_cls/blob/master/fig/Ins_f1_all_cmp_wide_mmm.png" width="900" height="330" />

#### Complexity Comparison
<img src="https://github.com/hchen605/ast_inst_cls/blob/master/fig/complexity_comparison.png" width="850" height="100" />
