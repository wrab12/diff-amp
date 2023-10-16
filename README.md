# Diff-AMP: An Integrated Tool for Antimicrobial Peptide Generation and Design Based on Diffusion and Reinforcement Learning Strategies

## Introduction
We have developed a unified AMP (Antimicrobial Peptide) design and optimization framework called Diff-AMP, which integrates various deep learning techniques to efficiently accomplish the tasks of AMP generation, evaluation, and optimization. As depicted in the following diagram, this framework encompasses the entire process of automatic molecule design and optimization, including AMP generation, evaluation, property prediction, and multi-objective iterative optimization. The framework comprises four modules, each incorporating advanced deep learning technologies. In Module A, we, for the first time, combine diffusion strategies and attention mechanisms (Module B) within a GAN network, proposing a novel AMP generation model to create AMPs that meet specific properties. Module C introduces large-parameter pre-trained models to acquire general knowledge about nodes and efficiently infer generated potential AMPs. Module D introduces a CNN model to perform multi-property prediction for AMPs. Module E marks the introduction of reinforcement learning strategies for the iterative optimization of generated AMPs.

![image](https://github.com/wrab12/diff-amp/blob/main/image/model.png)
## Environment
First, clone and navigate to the repository.
```bash
git clone https://github.com/jackrui/diff-amp
cd diff-amp
```
This process can take several minutes, depending on network speed.

Create and activate a virtual environment using python 3.9 with `virtualenv` or `conda`,
```python
# virtualenv (python 3.9)
virtualenv env
source env/bin/activate

# conda
conda create -n diff-amp python=3.9
conda activate diff-amp
```

Install dependencies and the local library with `pip`.
```bash

pip install -r requirements.txt

```
This process usually takes around 5 minutes.
## Datasets
- Find datasets for the generation model(Diff-RLGen), and the recognition model in the `data` directory.
- Download datasets for multi-property prediction from this [Google Drive link](https://drive.google.com/drive/folders/1ZAr3149wxE-362TsxjATwtdRVOPClk37?usp=drive_link).


## Usage
### generation model
- You can directly use our generation model through Hugging Face: [Diff-AMP Antimicrobial Peptide Generation](https://huggingface.co/spaces/jackrui/diff-amp-antimicrobial_peptide_generation)
- Train the generation model: Run `gan_diff.py`.
- Generate antimicrobial peptides: Use `gan_generate.py`. Get weight files from this [Google Drive link](https://drive.google.com/drive/folders/1vb_vvso29CQHMt43WpTGxoXTki16oNSm?usp=drive_link) and place them in the `weight` directory to use without retraining.
### classification model
- You can directly use our recognition model through Hugging Face: [Diff-AMP AMP Sequence Detector](https://huggingface.co/spaces/jackrui/diff-amp-AMP_Sequence_Detector)
- Train the classification model: Run `AMP_Classification.py`.
- If you want to directly identify your own peptides as antimicrobial, you can run `AMP_Classification_Prediction.py`. In this case, prepare a file named `seq.txt` with one sequence per line. You can modify the input format if needed. To use pre-trained weights, download them from this [Google Drive link](https://drive.google.com/drive/folders/1vb_vvso29CQHMt43WpTGxoXTki16oNSm?usp=drive_link) and place them in the `weight` directory.
### Multi-Attribute Prediction Model
- You can directly use our multi-attribute prediction model through Hugging Face: [Diff-AMP AMP Prediction Model](https://huggingface.co/spaces/jackrui/AMP_Prediction_Model)
- Perform property prediction on generated antimicrobial peptides: Run `predict.py`. Get the necessary weight files from this [Google Drive link](https://drive.google.com/drive/folders/1iLzwYbq0R3lwJum4laG1KshXs7oXD9fv?usp=drive_link) and place them in the 'models' directory.
## run
- Experience all steps of generation, recognition, and optimization.
```shell
run.sh
```

