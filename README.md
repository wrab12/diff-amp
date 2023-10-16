# Diff-AMP: An Integrated Tool for Antimicrobial Peptide Generation and Design Based on Diffusion and Reinforcement Learning Strategies

## Introduction
We have developed a unified AMP (Antimicrobial Peptide) design and optimization framework called Diff-AMP, which integrates various deep learning techniques to efficiently accomplish the tasks of AMP generation, evaluation, and optimization. As depicted in the following diagram, this framework encompasses the entire process of automatic molecule design and optimization, including AMP generation, evaluation, property prediction, and multi-objective iterative optimization. The framework comprises four modules, each incorporating advanced deep learning technologies. In Module A, we, for the first time, combine diffusion strategies and attention mechanisms (Module B) within a GAN network, proposing a novel AMP generation model to create AMPs that meet specific properties. Module C introduces large-parameter pre-trained models to acquire general knowledge about nodes and efficiently infer generated potential AMPs. Module D introduces a CNN model to perform multi-property prediction for AMPs. Module E marks the introduction of reinforcement learning strategies for the iterative optimization of generated AMPs.

![image](https://github.com/wrab12/diff-amp/blob/main/image/model.png)

## Datasets
- Find datasets for the generation model, Diff-RLGen, and the recognition model in the 'data' directory.
- Download datasets for multi-property prediction from this [Google Drive link](https://drive.google.com/drive/folders/1ZAr3149wxE-362TsxjATwtdRVOPClk37?usp=drive_link).

## Environment
- View all required dependencies in the 'requirements.txt' file.

## Usage
- Train the generation model: Run 'gan_diff.py'.
- Generate antimicrobial peptides: Use 'gan_generate.py'. Get weight files from this [Google Drive link](https://drive.google.com/drive/folders/1vb_vvso29CQHMt43WpTGxoXTki16oNSm?usp=drive_link) and place them in the 'weight' directory to use without retraining.
- Perform property prediction on generated antimicrobial peptides: Run 'predict.py'. Get the necessary weight files from this [Google Drive link](https://drive.google.com/drive/folders/1iLzwYbq0R3lwJum4laG1KshXs7oXD9fv?usp=drive_link) and place them in the 'models' directory.
- Experience all steps of generation, recognition, and optimization: Run 'run.sh'. This script initiates training from scratch.

Feel free to optimize the formatting and language to make this README document more aesthetically pleasing.
