# Batoonee - A Conversational Question Answering Assistant (Currently in development)

Batoonee uses Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for answering questions from a given context.
Batoonee is developed on Stanford's [COQA dataset](https://stanfordnlp.github.io/coqa/) 

1. The generator in the WGAN-GP is an attention based sequence-to-sequence network with bi-lstm encoder and lstm decoder. Attention used can be customized between luong and bahdanau attention. 

2. The discriminator or critic (as mentioned in WGAN paper) consists of question (context) encoder, answer encoder followed by additional layers to get the final score for the (question, answer) pair.
