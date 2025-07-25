
# **Finetune SigLIP2 Image Classification (Notebook)**


This notebook demonstrates how to fine-tune SigLIP 2, a robust multilingual vision-language model, for single-label image classification tasks. The fine-tuning process incorporates advanced techniques such as captioning-based pretraining, self-distillation, and masked prediction, unified within a streamlined training pipeline. The workflow supports datasets in both structured and unstructured forms, making it adaptable to various domains and resource levels.

---

| Notebook Name                        | Description                                      | Notebook Link |
|-------------------------------------|--------------------------------------------------|----------------|
| notebook-siglip2-finetune-type1  | Train/Test Splits  | [⬇️Download](https://huggingface.co/prithivMLmods/FineTuning-SigLIP2-Notebook/blob/main/Finetune-SigLIP2-Image-Classification/1.SigLIP2_Finetune_ImageClassification_TrainTest_Splits.ipynb) |
| notebook-siglip2-finetune-type2  | Only Train Split  | [⬇️Download](https://huggingface.co/prithivMLmods/FineTuning-SigLIP2-Notebook/blob/main/Finetune-SigLIP2-Image-Classification/2.SigLIP2_Finetune_ImageClassification_OnlyTrain_Splits.ipynb) |

> [!warning]
To avoid notebook loading errors, please download and use the notebook.
---

The notebook outlines two data handling scenarios. In the first, datasets include predefined train and test splits, enabling conventional supervised learning and generalization evaluation. In the second scenario, only a training split is available; in such cases, the training set is either partially reserved for validation or reused entirely for evaluation. This flexibility supports experimentation in constrained or domain-specific settings, where standard test annotations may not exist.

```
last updated : jul 2025
```
---




| **Type 1: Train/Test Splits** | **Type 2: Only Train Split** |
|------------------------------|------------------------------|
| ![Type 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/l0vfc0wtIp5mHgP-KGtff.png) | ![Type 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/xFXZYGbbL1PgoyyobVLym.png) |



---

| Platform | Link |
|----------|------|
| Huggingface Blog | [Blog](https://huggingface.co/blog/prithivMLmods/siglip2-finetune-image-classification) |
| GitHub Repository | [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PRITHIVSAKTHIUR/FineTuning-SigLIP-2) |
