# Probabilistic-Hadamard-U-Net

Objective: Probabilistic-Hadamard-U-Net is an end-to-end supervised learning approach for prostate MRI bias field correction.

Methods: At the start, a novel Hadamard U-Net (HU-Net) is introduced to extract the low-frequency scalar field, multiplied by the original input to obtain the prototypical corrected images. Specifically, HU-Net converts the input images from the time domain into the frequency domain via Hadamard transform. In the frequency domain, high-frequency components are eliminated using the trainable filter (scaling layer), hard-thresholding layer, and sparsity penalty. After that, a conditional variational autoencoder encodes possible bias field-corrected variants into a low-dimensional latent space. Random samples drawn from latent space are incorporated with a prototypical corrected image to generate multiple plausible images.

Results: Experimental results demonstrate the effectiveness of PHU-Net in correcting inhomogeneities with a fast inference speed. With the high quality corrected images from PHU-Net, the prostate MRI segmentation accuracy improves.

# Usage

python train_model.py

python test.py

Run the train_model.py to train the model and test.py to test the model.

# Dataset

We use 4 T2-weighted prostate MRI datasets in the experiment: 

HK dataset, UCL dataset, HCRUDB dataset  https://liuquande.github.io/SAML/


AWS dataset  http://medicaldecathlon.com/ .
