# Medical-Domain-Adaptation-with-Transformer

This repository contains implementaion of Transformer for ASR error correction and domain adaptation. Generally speaking, the deterioration in ASR performance can be caused by a domain mismatch between training and test data. Data domain is a specific topic characterizing the nature of the data, such as programming, finance, medicine, etc. This repository demonstrates how to fix domain mismatch errors using machine translation without making changes to the existing trained ASR model.

## Data preparation

- We tested the proposed approach on the medical data domain. 
- In order to implement the model, a huge amount of transcribed audio recordings of conversations with a medical context are required. Since the required amount of data is not available for free, the dataset were to synthesized from text.
- We used a [Kaggle Medical Transcriptions dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions), which contains samples of medical transcriptions.
- Then, the text samples were algorithmically cleaned and normalized so that a text-to-speech model could read them (see the text_normalization notebook in the data preparation directory).
- To synthesize audio samples, the implementation of the Voice Cloning system from the ["Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis"](https://arxiv.org/pdf/1806.04558.pdf) paper.
