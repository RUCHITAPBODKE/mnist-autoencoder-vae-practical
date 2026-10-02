# MNIST Autoencoder and Variational Autoencoder

**Practical No. 03 — Image Reconstruction and Generation**

This project implements and compares an Autoencoder (AE) and a Variational Autoencoder (VAE) on MNIST handwritten digits. It trains both models, evaluates reconstruction on held-out test images, plots original and reconstructed images, and generates example outputs.

## Contents

- `autoencoder_vae_practical.py` — complete, reproducible training and evaluation program.
- `Autoencoder_VAE_MNIST_Colab.ipynb` — Google Colab notebook to run the practical without setting up a local Python environment.
- `requirements.txt` — Python dependencies for running the script locally.
- `results/` — created after running; contains figures, metrics, downloaded dataset, and model weights. These generated files are excluded from Git.

## Dataset

MNIST contains 70,000 grayscale images of handwritten digits from 0 to 9. There are 60,000 training images and 10,000 test images. Each image is 28 × 28 pixels. The models use the image pixels as their targets; digit labels are not used for training. `ToTensor()` scales pixel values to [0, 1], matching the decoders' sigmoid output.

## Run in Google Colab

1. Create a GitHub repository and upload this project folder.
2. Open `Autoencoder_VAE_MNIST_Colab.ipynb` in GitHub and select **Open in Colab** (or in Colab choose **File → Open notebook → GitHub** and search for your repository).
3. Select **Runtime → Change runtime type → T4 GPU** if available. A GPU is optional.
4. Run the notebook cells from top to bottom. It downloads MNIST, trains both models, displays the figures and metrics, and offers result downloads.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python autoencoder_vae_practical.py --epochs 10 --batch-size 128
```

Set `--seed 42` to use the default seed explicitly. MNIST downloads on the first run. The script uses CUDA when available and otherwise runs on CPU.

## Models and objectives

Both models use a 16-dimensional latent representation. Their fully connected encoder and decoder sizes are `784 → 256 → 128 → 16` and `16 → 128 → 256 → 784`, respectively, with ReLU hidden activations and sigmoid pixel outputs.

The AE maps each image to a deterministic latent code and optimizes binary cross-entropy (BCE) reconstruction loss. It does not constrain its latent codes to follow a known distribution. Therefore the AE demonstration decodes codes from real training images; random standard-normal vectors are not guaranteed to produce meaningful outputs.

The VAE encoder predicts a mean and log variance for a Gaussian latent distribution. It samples using the reparameterization trick and minimizes reconstruction BCE plus `KL(q(z|x) || N(0,I))`. The KL term regularizes the latent space toward a standard normal prior, making prior sampling meaningful. Reconstructions are decoded from the posterior mean for stable evaluation; new samples are decoded from random prior vectors.

## Outputs and comparison

After execution, `results/` contains:

- `reconstructions.png` — test originals alongside AE and VAE reconstructions.
- `generated.png` — AE decodes of encoded training examples and VAE prior samples.
- `metrics.json` — seed, settings, training history, test MSE, and test BCE.
- `models.pt` — model weights and run configuration.
- `data/` — MNIST files downloaded by torchvision.

Compare images as well as metrics. The AE may produce sharper reconstructions because it focuses on reconstruction. The VAE's KL penalty can trade some pixel-level sharpness for a smoother, more organized latent space that supports sampling. Use the actual outputs from your run in the report; results depend on training settings and runtime.

## Viva concepts

- **Latent space:** the compact representation used by the decoder to reconstruct an input.
- **VAE KL divergence:** encourages encoded distributions to stay close to the standard normal prior.
- **Reparameterization:** writes a random latent sample as `z = μ + σ ⊙ ε`, where `ε ~ N(0,I)`, so gradients can train the encoder.
- **Why can a VAE sample new images?** Its latent distributions are regularized to match a known prior that can be sampled.
- **Why not sample arbitrary AE codes?** The AE objective does not train its codes to match a prior distribution.
- **Why may VAE outputs be blurrier?** Its reconstruction objective shares capacity with latent regularization, and pixelwise BCE does not directly optimize visual crispness.

## Reproducibility

The default seed is 42. The script sets Python, NumPy, and PyTorch seeds and uses deterministic cuDNN behavior. Exact equality can still vary across hardware and library versions. The notebook uses the same seed and model configuration.
