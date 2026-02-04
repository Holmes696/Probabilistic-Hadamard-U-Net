# Probabilistic-Hadamard-U-Net

![model](https://github.com/Holmes696/Probabilistic-Hadamard-U-Net/assets/162382272/dfa8ddec-292f-40c3-88d0-5ce37b6f9693)

Objective: Probabilistic-Hadamard-U-Net is an end-to-end supervised learning approach for prostate MRI bias field correction.

Methods: At the start, a novel Hadamard U-Net (HU-Net) is introduced to extract the low-frequency scalar field, multiplied by the original input to obtain the prototypical corrected images. Specifically, HU-Net converts the input images from the time domain into the frequency domain via Hadamard transform. In the frequency domain, high-frequency components are eliminated using the trainable filter (scaling layer), hard-thresholding layer, and sparsity penalty. After that, a conditional variational autoencoder encodes possible bias field-corrected variants into a low-dimensional latent space. Random samples drawn from latent space are incorporated with a prototypical corrected image to generate multiple plausible images.

Results: Experimental results demonstrate the effectiveness of PHU-Net in correcting inhomogeneities with a fast inference speed. With the high quality corrected images from PHU-Net, the prostate MRI segmentation accuracy improves.

# Variational Hadamard U-Net

<img width="1178" height="797" alt="image" src="https://github.com/user-attachments/assets/14c4028e-0d32-47c0-aca8-826a5f36d75b" />

Objective: Extend Probabilistic Hadamard U-Net (PHU-Net) framework to body MRI bias field correction.

Methods: The encoder comprises multiple convolutional Hadamard transform blocks (ConvHTBlocks), each integrating convolutional layers with a Hadamard transform (HT) layer. Specifically, the HT layer performs channel-wise frequency decomposition to isolate low-frequency components, while a subsequent scaling layer and semi-soft thresholding mechanism suppress redundant high-frequency noise. To compensate for the HT layer's inability to model inter-channel dependencies, the decoder incorporates an inverse HT-reconstructed transformer block, enabling global, frequency-aware attention for the recovery of spatially consistent bias fields. The stacked decoder ConvHTBlocks further enhance the capacity to reconstruct the underlying ground-truth bias field. Building on the principles of variational inference, we formulate a new evidence lower bound (ELBO) as the training objective, promoting sparsity in the latent space while ensuring accurate bias field estimation.

Results: Comprehensive experiments on body MRI datasets demonstrate the superiority of VHU-Net over existing state-of-the-art methods in terms of intensity uniformity, signal fidelity, and tissue contrast. Moreover, the corrected images yield substantial downstream improvements in segmentation accuracy. Our framework offers computational efficiency, interpretability, and robust performance across multi-center datasets, making it suitable for clinical deployment.

# Usage

Our approach integrates Hadamard transform layers into the U-Net architecture to enhance bias field correction performance. 

The folder "Hadamard_Transform_Layer_2D" contains the implementation of Hadamard transform layer. You need to import the function "WHTConv2D":

```python
from Hadamard_Transform_Layer_2D.WHT import WHTConv2D
```

For example, if the input tensor is 3x16x32x32 and the output is 3x16x32x32, single-path HT layer:

```python
WHTConv2D(32, 32, 16, 16, 1, residual=True)
```

3-path HT layer:

```python
WHTConv2D(32, 32, 16, 16, 3, residual=False)
```

The parameter "pod" in the function "WHTConv2D" stands for the number of paths.

# Dataset

We use 4 T2-weighted prostate MRI datasets in the experiment: 

HK dataset, UCL dataset, HCRUDB dataset  https://liuquande.github.io/SAML/

AWS dataset  http://medicaldecathlon.com/ .
