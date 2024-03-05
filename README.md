# Probabilistic-Hadamard-U-Net

![model](https://github.com/Holmes696/Probabilistic-Hadamard-U-Net/assets/162382272/6054792d-47ab-42a1-b5d6-94164125c76f)


Objective: NPPY is an end-to-end weakly supervised learning approach for converting raw head MRI images to intensity-normalized, skull-stripped brain in a standard coordinate space.

Methods: NPPY solves three sub-tasks simultaneously through a neural network, without individual sub-task supervision. The sub-tasks include geometric-preserving intensity mapping, spatial transformation, and skull stripping. The model disentangles intensity mapping and spatial normalization to solve the under-constrained objective.

Results: NPPY outperforms the state-of-the-art methods, which tackle only a single sub-task, according to quantitative results. The importance of NPP's architecture design is demonstrated through ablation experiments. Additionally, NPP provides users with the flexibility to control each task during inference time.
