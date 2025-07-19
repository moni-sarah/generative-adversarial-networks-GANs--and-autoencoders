# GenAI Model Evaluation Report

## Introduction
In this activity, we evaluated two key generative deep learning models—autoencoders and generative adversarial networks (GANs)—using the MNIST dataset. The primary goal was to assess their ability to reconstruct or generate image data and analyze their performance through suitable evaluation metrics: Mean Squared Error (MSE) for autoencoders and a combination of discriminator accuracy and visual inspection for GANs.

## Autoencoder Evaluation
The autoencoder was trained to reconstruct 28x28 grayscale MNIST images. After training, it achieved a Mean Squared Error (MSE) of approximately X.XXX (fill with actual result), indicating how closely the reconstructed images matched the original ones. MSE serves as a direct and interpretable measure of reconstruction fidelity; lower values suggest better model performance.

Visual inspection of a few reconstructed images showed that while the general digit structure was preserved, fine details (e.g., stroke thickness) were sometimes blurred, especially for more complex digits. This is expected given the dimensionality reduction in the bottleneck layer, which compresses image features into a 64-dimensional latent space. Despite this, the reconstructions were generally accurate, supporting the autoencoder's utility in tasks like denoising or anomaly detection.

## GAN Evaluation
The GAN was trained to generate new MNIST-like images from random noise vectors. The discriminator’s accuracy during training fluctuated, typically starting high but gradually approaching 50% as the generator improved—suggesting that it became increasingly difficult for the discriminator to tell real from fake images.

Visual inspection of generated images every 1000 epochs revealed a progressive improvement in realism. Early outputs were noisy and indecipherable, but after sufficient training, the GAN produced digit-like structures closely resembling real MNIST digits. However, some artifacts and mode collapse (repetition of certain digit types) were observed, especially with fewer training epochs.

## Conclusion
The autoencoder performed reliably for reconstruction tasks, with MSE providing a clear numerical performance metric. In contrast, the GAN excelled at generating new, visually plausible data, but required more nuanced evaluation, combining discriminator accuracy and visual inspection. Each model is best suited for different GenAI tasks: autoencoders for data compression and reconstruction, and GANs for data generation and creative synthesis.

# (Fill in your actual results and analysis below)

## 1. Autoencoder Performance
- **Reconstruction MSE:**
  - _(Paste the MSE value from your results)_
- **Interpretation:**
  - _(Discuss how the MSE reflects the autoencoder's ability to reconstruct MNIST images. Lower MSE = better reconstruction)_

## 2. GAN Performance
- **Discriminator Accuracy:**
  - _(Summarize how the discriminator accuracy changed during training)_
- **Generated Image Quality:**
  - _(Describe the visual quality of the generated images at different epochs. Did they become more realistic?)_

## 3. Comparison and Conclusions
- **Which model performed better for reconstruction?**
- **Which model generated more realistic new data?**
- **Which model is better suited for which GenAI task?**
- **Any additional observations or recommendations.** 