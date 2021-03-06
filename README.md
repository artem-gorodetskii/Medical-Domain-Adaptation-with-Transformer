# Medical-Domain-Adaptation-with-Transformer

This repository contains implementaion of Transformer for ASR error correction and domain adaptation. Generally speaking, the deterioration in ASR performance can be caused by a domain mismatch between training and test data. This repository demonstrates how to fix domain mismatch errors using machine translation without making changes to the existing trained ASR model.

## Data preparation

- We tested the proposed approach on the medical data domain. 
- In order to implement the model, a huge amount of transcribed audio recordings of conversations with a medical context are required. Since the required amount of data is not available for free, the dataset were synthesized from text.
- We used a [Kaggle Medical Transcriptions dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions), which contains samples of medical transcriptions.
- Then, the text samples were algorithmically cleaned and normalized so that a text-to-speech model could read them (see the text_normalization notebook in the data preparation directory).
- To synthesize audio samples, the modified implementation of the Voice Cloning system from the ["Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis"](https://arxiv.org/pdf/1806.04558.pdf) paper was used (see TTS in the data preparation directory).
- To produce transcripts the synthesized audio files were processed by [Vosk ASR model](https://alphacephei.com/vosk/models).
- Thus, a dataset composed of the ground truth and produced transcripts were obtained. In total, 134 hours of speech were synthesized, four of which were used for evaluation, and the rest for training. 

## Model
The model is based on a simplified version of the original model from the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. Here’s the list of differences:

- We use hidden dimension size 256 instead of 512;
- We employed learnable position embeddings instead of fixed static embeddings;
- The model has 3 Encoder and Decoder layers instead of 6;
- The internal dimension size of the Feed Forward layer is 512 instead of 2048;
- During training, we used a standard Adam optimizer with a constant learning rate without warm-up and cool-down steps;
- We apply Softmax without label smoothing (this technique tends to make the model less overconfident that results in improved generalization properties);
- The model has 14.7 million training parameters.

## Training
See the medical_domain_adaptation_training notebook for detail.

## Evaluation and Metrics
- Before using the Transformer model: WER = 20.52 %, BLEU = 70.4 %.
- After using the Transformer model: WER = 10.66 %, BLEU = 81.81 %.

See the medical_domain_adaptation_inference notebook for detail.

## Conclusions
- Correcting transformers can improve the accuracy of speech recognition more than two times.
- Transformer corrects the ASR errors for the medical domain, bringing the WER closer to the in-domain level (7-8%).

