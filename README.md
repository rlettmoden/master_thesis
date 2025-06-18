# MultiGFM

**MultiGFM** is a novel framework for integrating Satellite Image Time Series (SITS) into mono-temporal multi-modal Geospatial Foundation Models (GFMs) for land use and land cover (LULC) mapping. This approach enhances temporal learning in multi-modal remote sensing pipelines while keeping computational complexity manageable.

[Link to Paper](https://www.dgpf.de/src/tagung/jt2025/proceedings/paper/10_3LT2025_Lettmoden_et_al.pdf)

## Key Features

- ✅ Incorporates temporal patterns using SITS with optical and radar data.
- ✅ Utilizes a shared spatial encoder (DOFA) with modality-specific decoders.
- ✅ Uses lightweight temporal encoders (L-TAE, temporal max-pooling).
- ✅ Achieves **82.4% mIoU** on LULC segmentation in the Harz region, Germany.

## Custom Harz DataSet

- **Sentinel-1 (Radar)**: 10m resolution, 51 images (VV, VH polarization)
- **Sentinel-2 (Optical)**: 10m resolution, 7 cloud-free images
- **PlanetScope (Optical)**: 3m resolution, 7 images (RGB + NIR)

Labels derived from:
- ESA WorldCover Maps (2020 & 2021)
- Tree species map (2018)
- Manual annotations for dead trees

## Architecture Overview

MultiGFM processes multi-modal SITS through the following steps:

1. **Preprocessing**: Normalize, patchify, split into train/val/test.
2. **Shared Spatial Encoder**: Use DOFA's ViT-based encoder.
3. **Spatial Decoders**: Prithvi (light-weight), but also SegFormer and UNet implemented
4. **Temporal Encoding**:
   - Long sequences: L-TAE
   - Short sequences: Temporal max-pooling
5. **Fusion**: 
    - Early Fusion: Most efficient, but more 
    - Late Fusion: Concatenate spatial-temporal features.
6. **Segmentation Head**: Predict LULC classes.

![Framework Overview](./docs/MultiGFM_Framework.png) *(Replace with actual image if available)*

## Experimental Highlights

| Component             | Best Option            | Notes                                 |
|----------------------|------------------------|----------------------------------------|
| Spatial Decoder       | Prithvi                | Lightweight, better than UPerNet       |
| Fusion Method         | Late Fusion (Concat)   | Allows feature learning across modalities |
| Temporal Encoder      | L-TAE or Max-Pooling    | Based on sequence length               |
| Regularization        | Data Aug + Temp Dropout | Reduced overfitting significantly      |
| Init Weights          | DOFA pre-trained       | Outperforms ImageNet and scratch       |

## Requirements

Machine with multiple GPUs to enable (synchronized) batchnorm greater 1. 

## Performance

- **Best Test mIoU**: 82.4%
- **Params**: ~132M
- **FLOPs**: ~448 GFLOPs

## Setup

1. Clone the repo
2. Install dependencies (GeoSeg, PyTorch Lightning, fvcore, etc.)
3. Prepare the dataset by calculating norm, splitting patches, checking augmentations
4. Run training script with your desired config and dataset(see)

## Citation

If you use MultiGFM, please cite:

```
@inproceedings{lettmoden2025multigfm,
title={MultiGFM: multi-temporal framework for multi-modal geospatial foundation models},
author={Lettmoden, Reiko and Achanccaray, Pedro and Bittner, Ksenia and Gerke, Markus},
booktitle={Dreiländertagung der DGPF, der OVG und der SGPF, Band 33},
year={2025}
}
```

## Acknowledgments

This work was supported by the EXDIMUM project, funded by the German Federal Ministry of Education and Research (BMBF) as part of the WaX (Water Extreme Events) program under the FONA strategy.

