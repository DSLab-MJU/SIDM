# Semantic Interpolative Diffusion Model: Bridging the Interpolation to Masks and Colonoscopy Image Synthesis for Robust Generalization (SIDM)

This is the official implementation of 'Semantic Interpolative Diffusion Model: Bridging the Interpolation to Masks and Colonoscopy Image Synthesis for Robust Generalization' at **MICCAI 2025**.
<p align="center">
<img src=/assets/result.png />
</p>

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Your Own SIDM](#training-your-own-sidm)
- [Sampling with SIDM](#sampling-with-sidm)
- [Inference with SIDM](#inference-with-sidm)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)

## Requirements
```bash
conda create -n SIDM python=3.8.10
conda activate SIDM
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## Dataset Preparation
The proposed framework requires different processing for medical video data and snapshot data; therefore, a separation is necessary.

Please organize the dataset with the following structure:
```angular2
├── ${data_root}
│ ├── ${train_dataset_dir}
│ │ ├── images_video
│ │ │ ├── ***.png
│ │ ├── images
│ │ │ ├── ***.png
│ │ ├── masks_video
│ │ │ ├── ***.png
│ │ ├── masks
│ │ │ ├── ***.png

```

Details on the processing of the proposed background semantic labels can be found in ```datasets_label.log```, which is based on [CVCClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/).

## Training Your Own SIDM
To train your own SIDM, follow these steps:

1. Verify whether the training dataset, if it is a video dataset, has been properly separated.
2. Verify the data processing procedure by referring to ```polyp.py```.
3. Run the following command:

```bash
python train.py --data_path ./TrainDataset \
               --save_dir 'your_path' \
               --image_size 256 \
               --n_epoch 5000 \
               --n_T 1000 \
               --batch_size 2 \

```
## Sampling with SIDM
To sampling with SIDM, run the following command:

```bash
python sampling.py
```

**Sampling**

You can configure the interpolation ratio within the code to control the sampling process.
By default, a 1:1 sampling ratio is used.

**Note**

Make sure to correctly set the ```save_dir``` to avoid file saving issues.

## Inference with SIDM
The inference code applies interpolation between any two desired data samples.

We provide the ```LabeledDataset``` used in this study.
To perform inference using this dataset, please refer to ```inference.ipynb```.

## Acknowledgement
This repository is based on [LDM](https://github.com/CompVis/latent-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), [ArSDM](https://github.com/DuYooho/ArSDM), [CFG](https://github.com/TeaPearce/Conditional_Diffusion_MNIST) and [SDM](https://github.com/WeilunWang/semantic-diffusion-model). We sincerely thank the original authors for their valuable contributions and outstanding work.


## Citations
```angular2
@InProceedings{HeoCha_Semantic_MICCAI2025,
        author = { Heo, Chanyeong and Jung, Jaehee},
        title = { { Semantic Interpolative Diffusion Model: Bridging the Interpolation to Masks and Colonoscopy Image Synthesis for Robust Generalization } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15970},
        month = {September},
        page = {519 -- 529}
}
```
