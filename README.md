# Deep Neural Network Based Crowd Density Estimation

This project aims to estimate crowd density using only deep neural networks, without relying on any other detection methods. In the planning of this project, the refactoring of the code of the existing thesis method will be prioritized first, and then a new network architecture will be attempted to be used for density estimation

</br>

## Features

* 📕Refactor the code to improve readability
* 📈Visualization of results, output of images with bracketing and density maps
* 🛠️Structural harmonization, where only the network architecture is changed in the different approaches, while the rest remains unchanged
* 🚀[PLAN] Planning to refactor using Rust
* 💻[PLAN] Planning to design a real-time crowd estimation application

</br>

## Install

There are no special environmental requirements for this project, test with:

* `Ubuntu 22.04 | CUDA-11.8 | Pytorch-2.0.0`

* `Windows 11 | CUDA-12.4 | Pytorch-2.4.0`

* [PLAN] Support for only-cpu in the future

</br>

## Dataset

This project use [Kaggle-ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech) with **Part-B**, your file structure should be:  

```bash
Crowd-Density-Estimation
├─dataset
│  └─ShanghaiTech_Crowd_Counting_Dataset
│      ├─part_A_final
│      │  ├─test_data
│      │  │  ├─ground_truth
│      │  │  └─images
│      │  └─train_data
│      │      ├─ground_truth
│      │      └─images
│      └─part_B_final
│          ├─test_data
│          │  ├─ground_truth
│          │  └─images
│          └─train_data
│              ├─ground_truth
│              └─images
├─models
└─result
    ├─ckpt
    ├─density
    └─images
```

</br>

## Train

The hyperparameters to be used for training are all set in `config.py`

```bash
python main.py --mode train

# or train&valid
python main.py --mode both
```

</br>

## Valid

The hyperparameters to be used for validate are all set in `config.py`

In this step, a density map and an RGB image with bounding boxes will be generated

* density map: `./result/density/[RUN_DATE]/`
* rgb with boxes: `./result/image/[RUN_DATE]/`

```bash
python main.py --mode test

# or train&valid
python main.py --mode both
```

</br>

## Experiments

| Network                    | Best MAE ↓ | Epoch          |
| -------------------------- | ---------- | -------------- |
| CAN                        | 9.329      | 100            |
| CAN(net structure revised) | 15.315     | 100(less time) |
| P2P-Net                    |            |                |
|                            |            |                |
|                            |            |                |

</br>

## References

[Context-Aware Crowd Counting](https://arxiv.org/abs/1811.10452)</br>
[Rethinking Counting and Localization in Crowds:A Purely Point-Based Framework](https://arxiv.org/abs/2107.12746)

