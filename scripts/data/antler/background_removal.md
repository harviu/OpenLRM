# Background Removal 

## Overview

This document describes the background removal pipeline for antler preprocessing including method, results, and limitations discovered through implementation.

## Method

### Approach: Enhanced rembg with COLMAP guidance

We implemented a three stage background removal pipeline

1. Two Pass rembg Processing
    * Pass 1: Standard segmentation with conservative parameters
    * Pass 2: Aggresive erosion targeting objects in contact (e.g. people behind antler)
2. COLMAP 3D Spacial Guidance
    * Leverage reconstructed 3D point cloud to indentify antler location
    * Create spacial mask around projected antler points
    * Contrain background removal to antler region only
3. Statistical Outlier Detection
    * Gaussian method: Flag images with Z-score > 2.5σ from mean foreground area
    * Indentify failed segmentations where background objects still remain

### Parameters

Standard Processing (Pass 1)
```
alpha_matting_foreground_threshold: 240-285

alpha_matting_background_threshold: 10-40

alpha_matting_erode_size: 10-30

expansion_ratio (COLMAP): 1.2-1.5
```

Aggressive Processing (Pass 2)
```
alpha_matting_foreground_threshold: 285

alpha_matting_background_threshold: 40

alpha_matting_erode_size: 25

expansion_ratio (COLMAP): 1.2
```

## Results

### Quantitative Performance

| Metric | Result |
|--------|--------|
| Total images processed | 520 (13 antlers, ~40 images each) |
| Success rate (clean backgrounds) | 92% |
| Outliers detected per antler | 3-8 images (5-15%) |
| Human-antler contact failure rate | 35% |
| Processing time per image (GPU) | 2-3 minutes |

### What Worked Well

**Non touching background objects**: Chairs, tables, picture frames, whiteboards removed successfully

**Outlier detection**: Statistical analysis effectively indentified problematic results requiring manual review

**COLMAP guidance**: Spacial contraints reduced false positives in cluttered images

**Edge preservation**: Alpha matting maintained fine antler details (tines, texture)

### Example Outlier Statistics

Mean area: ____ pixels

Median area: _____ pixels

Std deviation: _____ pixels

Outliers detected: ___

    Image __: ____ pixels ()
    Image ___: ___ pixels ()

## Limitations

### Known Failure Cases

*   Human to antler physical contact
    * When people hold/touch antlers, rembg treats them as a single connected object
    * Aggresive erosion removes edges but not center mass
    * No sematic understanding to distinguish person from antler

*   Partial person removal artifacts 
    * People often partially removed, leaving visble remnants
    * Two pass processing helps but insufficient for contact scenarios

*   Outlier detection scope
    * Only flags problems, does not automatically correct them
    * Requires manual review or alternative segmentation method for flagged images

### Root Cause Analysis

Technical Limitation: rembg uses saliency based segmentation

* Segments "visually interesting" regions without object understanding
* Cannot distinguish between person and antler categories
* Fails when objects are spacially connected in the image (e.g. human and antler)

## Recommendations

### For Current Dataset
1. Use rembg + COLMAP for all images
2. Inspect flagged outliers
3. Remove outliers from training set if < 10% of total data

### For Future Datasets
1. Minimize human and antler contact during photography
2. Ensure clear spacial seperation between antler and background objects
3. Consider capture methods in a less noisy environment

## Usage

### Running the Pipeline

Process antlers with background removal and outlier detection:

``` sbatch run_all_antlers.sbatch ```

### Output Structure
```
antler_training_data/
├── train_uids.json
├── val_uids.json       
└── antler_X/
    ├── intrinsics.npy       # Camera intrinsics
    ├── pose/                # Camera to world poses
    │   └── 000.npy
    ├── rgba/                # Background removed images
    │   └── 000.png 
    └── outlier_report.txt   # Statistical analysis
```

### Review Results

Check the outlier reports for each antler where # is the antler ID

```
# View outlier detection results
cat antler_training_data/antler_#/outlier_report.txt

# Inspect flagged images manually
ls antler_training_data/antler_#/rgba/
```

## References

*   rembg: https://github.com/danielgatis/rembg
*   COLMAP: https://colmap.github.io/
