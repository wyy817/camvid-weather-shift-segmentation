# CamVid Weather-Shift Segmentation: Small-Object Study

Group project for XJTLU DTS304TC (Machine Learning), May 2026: semantic segmentation of CamVid street scenes (14 classes) with a U-Net, studying how a model trained on sunny images behaves under a weather shift (cloudy) and on small objects. 街景语义分割：晴天模型迁移到阴天，以及小目标问题。

## Team project note
Group 73 (4 members). This repository contains **my individual part only**: the small-object study (Part C) in `small_object_study.ipynb`. The group parts (pretrained-model evaluation on sunny data and transfer to the cloudy domain) were done jointly and are not included.

## Group-level results (for context)
Transfer to cloudy images (74 training images): validation mIoU rose from 0.351 (no fine-tuning) to 0.621 with the final recipe (CE + Dice loss, LR scheduler, AdamW).

## My part: small objects (Car_Bus and Pedestrian)
- **Problem.** The baseline segments large objects well but small ones poorly, a large object-level F1 gap for Car_Bus (large vs small) and very low small-pedestrian performance.
- **Method.** Four from-scratch U-Net ablations on the sunny split: (0) plain CE, (1) class-weighted CE, (2) class-weighted CE + Dice, (3) final method: class-weighted CE + Dice + small-object zoom-in (crop-and-enlarge) augmentation. Object-level (large vs small) statistics, qualitative failure cases, and a single final evaluation on the held-out test split.
- **Results (validation unless stated).**
  - Pedestrian IoU: 0.029 -> 0.269
  - Car_Bus IoU: 0.562 -> 0.715
  - Over the four ablations, mIoU 0.411 -> 0.481; test mIoU 0.481 (consistent with validation, so no sign of over-fitting to the validation set).
  - Failure analysis: class weighting alone raised Pedestrian false positives from 271 to 957, which is why the Dice term was added.

## Notebook structure
Setup and focus classes (assigned from the student-ID digit: Car_Bus, Pedestrian) -> C1 baseline small-object analysis and failure cases -> C2 ablation training (loss functions, class weights, zoom-in augmentation) -> C3 validation comparison, one-off test evaluation, side-by-side qualitative comparison.

## Running it
The notebook relies on course-provided helper code and data that are **not included**: the CamVid dataset (sunny/cloudy splits), a pretrained sunny-domain model, and helper modules used for data loading / augmentation / U-Net / object-level metrics. I have not re-run it for this repository; outputs and figures were stripped, so re-run to regenerate them.

## Tech stack
Python, PyTorch, U-Net, Matplotlib.
