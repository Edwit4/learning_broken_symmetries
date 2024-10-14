### Learning Broken Symmetries with Resimulation and Encouraged Invariance

Authors: Edmund Witkowski, Daniel Whiteson

This repository holds code to accompany the paper ["Learning Broken Symmetries with Resimulation and Encouraged Invariance"](https://arxiv.org/abs/2311.05952).

The relevant dataset may be found on [Zenodo](https://zenodo.org/records/13921968).

## Organization

* The data obtained from the above Zenodo link is to be stored in src/data.

* Figures are stored in the src/figures directory. These are provided in the repo, but may be generated using the scripts in src/figure_generation.

* The src/utilities directory contains necessary functionality for the scripts in the other directories.

* Cross validation and bootstrapping scripts are found in the src/experiments folder. The "fulluniform" and "fullrect" directory prefixes denote whether the scripts refer to the uniform or rectangular pixelation cases, respectively. The names of the subdirectories within these denote model architecture and learning approach. "fcn" refers to a basic fully connected network, while "pfn" refers to a particle flow network. Directories named with "no_aug" mean that no augmented image variants are included in the training, while "with_aug" means these will be included but no symmetry term is included in the loss. Directories with "sym_enc" do include the symmetry term. Finally, directories ending with "pix" will use augmented variants where the symmetry transformations are applied to the pixels of an image after the simulated detector response (corresponding directories lacking "pix" use images where the transformation is applied pre-detector response.)

## Full Instructions

1. Download the dataset (found at the above Zenodo link), and place it in src/data.

2. Run scan_params.py followed by scan_summary.py in each of the src/experiments/\*cross_validation sub-directories.

3. Run train_size_scan.py in each of the src/experiments/\*bootstrapping sub-directories.

4. Run the scripts in src/figure_generation to obtain the final set of figures.
