# VQVAE
Simple PyTorch implementation of VQ-VAE: https://arxiv.org/pdf/1711.00937.

## Usage

1) Execute ```python vqvae_train.py``` to train a VQ-VAE for CIFAR-10.
2) Execute ```python vqvae_recon.py``` to test the reconstruction capability of the model.
3) Execute ```python train_prior.py``` to train a PixelCNN model to learn a prior $p(z|\text{class})$, where class is an inger between [0,9] (CIFAR-10 classes).
4) Execute ```python test_prior.py``` to test resulting  model for image generation (the class is set to "ship", but you can change it). The results are saved: ```output_gen.png```.

## Results
Before prior training:

![output](https://github.com/user-attachments/assets/edf0ccbe-84b0-4481-8b8e-22baf320dd28)

After learning prior (examples for "ship" class):

 ![output_gen](https://github.com/user-attachments/assets/2799bccb-ce21-4ae1-aedc-f6b6295661b1)

## Thanks

https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
https://github.com/ritheshkumar95/pytorch-vqvae



