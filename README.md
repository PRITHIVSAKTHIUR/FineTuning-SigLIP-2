> [!important]
Blog: https://huggingface.co/blog/prithivMLmods/siglip2-finetune-image-classification 

## Fine-tuning Notebooks

| Notebook Name                        | Description                                      | Download Link |
|-------------------------------------|--------------------------------------------------|----------------|
| `notebook-siglip2-finetune-type1`  | Fine-tune and evaluate using `full_train` data  | [Download](https://github.com/PRITHIVSAKTHIUR/FineTuning-SigLIP-2/blob/main/notebook-siglip2-finetune-type1/siglip2-train-and-evaluate-using-full-train-data.ipynb) |
| `notebook-siglip2-finetune-type2`  | Fine-tune and evaluate using `test` data        | [Download](https://github.com/PRITHIVSAKTHIUR/FineTuning-SigLIP-2/blob/main/notebook-siglip2-finetune-type2/siglip2-tarin-and-evaluate-using-test-data.ipynb) |



# Fine-Tuning SigLIP 2 for Multi-Label Image Classification

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ayzPksY-A8Do5HHpU2xEM.png)

- SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

- SigLIP 2 introduces new multilingual vision-language encoders that build on the success of the original SigLIP. In this second iteration, we extend the original image-text training objective by integrating several independently developed techniques into a unified approach. These include captioning-based pretraining, self-supervised losses such as self-distillation and masked prediction.  

- The script below is used for fine-tuning SigLIP 2 foundational models on a single-label image classification problem.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/qRoNqE-IqaNo3cq0959Fu.png)

---
## Fine-Tuning >

## 1. Installing Required Packages

We start by installing all necessary packages. These include libraries for evaluation, data handling, model training, image processing, and additional utilities. If you’re running this in Google Colab, you might skip some installations.

```python
!pip install -q evaluate datasets accelerate
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q huggingface_hub
```

```python
!pip install -q imbalanced-learn
#Skip the installation if your runtime is in Google Colab notebooks.
```

```python
!pip install -q numpy
#Skip the installation if your runtime is in Google Colab notebooks.
```
```python
!pip install -q pillow==11.0.0
#Skip the installation if your runtime is in Google Colab notebooks.
```
```python
!pip install -q torchvision
#Skip the installation if your runtime is in Google Colab notebooks.
```
```python
!pip install -q matplotlib
!pip install -q scikit-learn
#Skip the installation if your runtime is in Google Colab notebooks.
```

*Explanation:*  
These commands install libraries such as `evaluate`, `datasets`, `transformers`, and `huggingface_hub` for model training and evaluation; `imbalanced-learn` for handling imbalanced datasets; and other common libraries like `numpy`, `pillow`, `torchvision`, `matplotlib`, and `scikit-learn`.

---

## 2. Importing Libraries and Configuring Warnings

Next, we import standard libraries and configure warnings to keep the output clean.

```python
import warnings
warnings.filterwarnings("ignore")
```

*Explanation:*  
We import the `warnings` module and set it to ignore warnings so that our notebook output remains uncluttered.

---

## 3. Additional Imports

Here we import modules required for data manipulation, model training, and image preprocessing.

```python
import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
#....................................................................

#Retain this part if you are working on ViTForImageClassification.
    ViTImageProcessor,
    ViTForImageClassification,

#....................................................................

    DefaultDataCollator
)
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)
```

*Explanation:*  
This block brings in various libraries and functions:
- **Data processing:** `numpy`, `pandas`, and `itertools`
- **Visualization:** `matplotlib.pyplot`
- **Metrics:** Accuracy, F1 score, and confusion matrix from scikit-learn
- **Oversampling:** `RandomOverSampler` to balance classes
- **Datasets and Transformers:** For loading datasets and model training
- **Torch and torchvision:** For tensor handling and image transformations

---

## 4. Handling Image Metadata and Truncated Images

For image processing outside Colab, we import additional modules to handle image metadata and enable the loading of truncated images.

```python
#.......................................................................

#Retain this part if you're working outside Google Colab notebooks.
from PIL import Image, ExifTags

#.......................................................................

from PIL import Image as PILImage
from PIL import ImageFile
# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

*Explanation:*  
This section uses the Python Imaging Library (PIL) to handle images. Enabling the loading of truncated images ensures that the script does not crash when encountering slightly corrupted image files.

---

## 5. Loading and Preparing the Dataset

We load a pre‐defined dataset and extract file paths and labels into lists. Then we create a DataFrame for further processing.

```python
from datasets import load_dataset
dataset = load_dataset("--your--dataset--goes--here--", split="train")

from pathlib import Path

file_names = []
labels = []

for example in dataset:
    file_path = str(example['image'])
    label = example['label']

    file_names.append(file_path)
    labels.append(label)

print(len(file_names), len(labels))
```

*Explanation:*  
- The dataset is loaded using Hugging Face’s `load_dataset` function.
- We iterate over the dataset to extract image file paths and labels.
- Finally, we print the number of images and labels to verify the extraction.

---

## 6. Creating a DataFrame and Balancing the Dataset

The next step converts the lists into a Pandas DataFrame and balances the classes using oversampling.

```python
df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
print(df.shape)

df.head()
df['label'].unique()

y = df[['label']]
df = df.drop(['label'], axis=1)
ros = RandomOverSampler(random_state=83)
df, y_resampled = ros.fit_resample(df, y)
del y
df['label'] = y_resampled
del y_resampled
gc.collect()
```

*Explanation:*  
- A DataFrame is created from the file names and labels.
- We inspect the DataFrame shape, head, and unique labels.
- To handle imbalanced classes, the `RandomOverSampler` is used to balance the dataset.
- Garbage collection is called to free up unused memory.

---

## 7. Inspecting Dataset Images

We check a couple of images from the dataset to ensure that they are loaded correctly.

```python
dataset[0]["image"]
dataset[9999]["image"]
```

*Explanation:*  
This simple check confirms that images can be accessed by their index, ensuring the dataset is loaded correctly.

---

## 8. Working with a Subset of Labels

We print a subset of labels and then define the complete list of labels used for classification.

```python
labels_subset = labels[:5]
print(labels_subset)

labels_list = ['example_label_1', 'example_label_2']

label2id, id2label = {}, {}
for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)
```

*Explanation:*  
- A small subset of labels is printed to give a glimpse of the data.
- The `labels_list` is defined for a single-label classification problem.
- Two dictionaries (`label2id` and `id2label`) map labels to numeric IDs and vice versa.
- A `ClassLabel` object is created to standardize the label format for the dataset.

---

## 9. Mapping and Casting Labels

We convert string labels to integer values and cast the dataset’s label column.

```python
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)
```

*Explanation:*  
- The `map_label2id` function converts label strings to integers.
- The `map` function applies this conversion to the entire dataset.
- Finally, the label column is cast using the `ClassLabel` object for consistency.

---

## 10. Splitting the Dataset

The dataset is split into training and testing subsets with a 60/40 ratio, while maintaining class stratification.

```python
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")
train_data = dataset['train']
test_data = dataset['test']
```

*Explanation:*  
- The dataset is split into training and test sets.
- Stratification ensures that the proportion of classes remains consistent in both splits.

---

## 11. Setting Up the Model and Processor

We load the SigLIP2 model and its corresponding image processor. The processor is used to extract preprocessing parameters for the images.

```python
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification

# Use AutoImageProcessor instead of AutoProcessor
model_str = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_str)

# Extract preprocessing parameters
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
```

*Explanation:*  
- The SigLIP2 model is loaded from Hugging Face using its model identifier.
- The `AutoImageProcessor` retrieves the preprocessing configuration (mean, standard deviation, and image size) required for input normalization.

---

## 12. Defining Data Transformations

We define training and validation image transformations using `torchvision.transforms`. These include resizing, random rotations, sharpness adjustments, and normalization.

```python
# Define training transformations
_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(90),
    RandomAdjustSharpness(2),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

# Define validation transformations
_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])
```

*Explanation:*  
- The training transformations add data augmentation (rotation and sharpness adjustment) to improve model generalization.
- The validation transformations only resize and normalize the images, ensuring consistency during evaluation.

---

## 13. Applying Transformations to the Dataset

Functions are defined to apply the above transformations to the dataset, and these functions are then set as the transformation functions for the training and testing datasets.

```python
# Apply transformations to dataset
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Assuming train_data and test_data are loaded datasets
train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)
```

*Explanation:*  
- The `train_transforms` and `val_transforms` functions convert images to RGB and apply the respective transformations.
- These functions are then set to the training and test datasets so that each image is preprocessed on-the-fly during training and evaluation.

---

## 14. Creating a Data Collator

A custom collate function is defined to prepare batches during training by stacking images and labels into tensors.

```python
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

*Explanation:*  
This function gathers individual examples into a batch by stacking the processed image tensors and converting the labels to a tensor. It is passed to the Trainer to ensure proper batching.

---

## 15. Initializing the Model

We load the SigLIP2 model for image classification, configure the label mappings, and print the number of trainable parameters.

```python
model = SiglipForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))
model.config.id2label = id2label
model.config.label2id = label2id

print(model.num_parameters(only_trainable=True) / 1e6)
```

*Explanation:*  
- The SigLIP2 model is instantiated for image classification with the specified number of classes.
- The label mappings are assigned to the model configuration.
- The number of trainable parameters (in millions) is printed to give an idea of the model’s size.

---

## 16. Defining Metrics and the Compute Function

We load an accuracy metric and define a function to compute accuracy during evaluation.

```python
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {
        "accuracy": acc_score
    }
```

*Explanation:*  
- The `evaluate` library is used to load an accuracy metric.
- The `compute_metrics` function calculates the accuracy by comparing predicted labels with the true labels.

---

## 17. Setting Up Training Arguments

Training parameters are defined using the `TrainingArguments` class. These include batch sizes, learning rate, number of epochs, logging details, and more.

```python
args = TrainingArguments(
    output_dir="siglip2-finetune",
    logging_dir='./logs',
    evaluation_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)
```

*Explanation:*  
These arguments configure:
- The output directory and logging.
- The evaluation strategy (evaluating after each epoch).
- The learning rate, batch sizes, number of training epochs, and weight decay.
- Settings to load the best model at the end and limit checkpoint storage.

---

## 18. Initializing the Trainer

We now initialize the Hugging Face Trainer with the model, training arguments, datasets, data collator, metrics function, and image processor (used as the tokenizer).

```python
trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```

*Explanation:*  
The `Trainer` is the main interface for training and evaluation. All necessary components (model, arguments, datasets, collate function, and metrics) are passed to it.

---

## 19. Evaluating, Training, and Predicting

Before training, we evaluate the model on the test set. Then, training is started, followed by a second evaluation and obtaining predictions.

```python
trainer.evaluate()

trainer.train()

trainer.evaluate()

outputs = trainer.predict(test_data)
print(outputs.metrics)
```

*Explanation:*  
- The initial evaluation provides a baseline before fine-tuning.
- After training, the model is evaluated again.
- Predictions on the test set are obtained, and the resulting metrics are printed.

---

## 20. Computing Additional Metrics and Plotting the Confusion Matrix

We compute accuracy and F1 scores, plot a confusion matrix (if there are few classes), and print a full classification report.

```python
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):

    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) <= 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

print()
print("Classification report:")
print()
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))
```

*Explanation:*  
- Predictions are compared against true labels to calculate accuracy and macro F1 scores.
- A custom function plots the confusion matrix if the number of classes is small.
- Finally, a detailed classification report is printed.

---

## 21. Saving the Model and Uploading to Hugging Face Hub

The fine-tuned model is saved locally and then uploaded to the Hugging Face Hub.

```python
trainer.save_model()

#upload to hub
from huggingface_hub import notebook_login
notebook_login()

from huggingface_hub import HfApi

api = HfApi()
repo_id = f"prithivMLmods/siglip2-finetune"

try:
    api.create_repo(repo_id)
    print(f"Repo {repo_id} created")

except:

    print(f"Repo {repo_id} already exists")

api.upload_folder(
    folder_path="siglip2-finetune/",
    path_in_repo=".",
    repo_id=repo_id,
    repo_type="model",
    revision="main"
)
```

*Explanation:*  
- The `trainer.save_model()` call saves the best fine-tuned model.
- The Hugging Face Hub login is initiated with `notebook_login()`.
- The repository is created (or verified to exist) using the `HfApi`.
- Finally, the model folder is uploaded to the repository.

---

## Final Script

Below is the full script in one place without the dashed-line separators:

```python
!pip install -q evaluate datasets accelerate
!pip install -q transformers
!pip install -q huggingface_hub

!pip install -q imbalanced-learn
#Skip the installation if your runtime is in Google Colab notebooks.

!pip install -q numpy
#Skip the installation if your runtime is in Google Colab notebooks.

!pip install -q pillow==11.0.0
#Skip the installation if your runtime is in Google Colab notebooks.

!pip install -q torchvision
#Skip the installation if your runtime is in Google Colab notebooks.

!pip install -q matplotlib
!pip install -q scikit-learn
#Skip the installation if your runtime is in Google Colab notebooks.

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from PIL import Image, ExifTags
from PIL import Image as PILImage
from PIL import ImageFile
# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import load_dataset
dataset = load_dataset("--your--dataset--goes--here--", split="train")

from pathlib import Path

file_names = []
labels = []

for example in dataset:
    file_path = str(example['image'])
    label = example['label']

    file_names.append(file_path)
    labels.append(label)

print(len(file_names), len(labels))

df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
print(df.shape)

df.head()
df['label'].unique()

y = df[['label']]
df = df.drop(['label'], axis=1)
ros = RandomOverSampler(random_state=83)
df, y_resampled = ros.fit_resample(df, y)
del y
df['label'] = y_resampled
del y_resampled
gc.collect()

dataset[0]["image"]
dataset[9999]["image"]

labels_subset = labels[:5]
print(labels_subset)

labels_list = ['example_label_1', 'example_label_2']

label2id, id2label = {}, {}
for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

print("Mapping of IDs to Labels:", id2label, '\n')
print("Mapping of Labels to IDs:", label2id)

def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")

train_data = dataset['train']
test_data = dataset['test']

from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification

# Use AutoImageProcessor instead of AutoProcessor
model_str = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_str)

# Extract preprocessing parameters
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Define training transformations
_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(90),
    RandomAdjustSharpness(2),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

# Define validation transformations
_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

# Apply transformations to dataset
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Assuming train_data and test_data are loaded datasets
train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = SiglipForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))
model.config.id2label = id2label
model.config.label2id = label2id

print(model.num_parameters(only_trainable=True) / 1e6)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {
        "accuracy": acc_score
    }

args = TrainingArguments(
    output_dir="siglip2-finetune",
    logging_dir='./logs',
    evaluation_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.evaluate()

trainer.train()

trainer.evaluate()

outputs = trainer.predict(test_data)
print(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):

    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) <= 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

print()
print("Classification report:")
print()
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

trainer.save_model()

#upload to hub
from huggingface_hub import notebook_login
notebook_login()

from huggingface_hub import HfApi

api = HfApi()
repo_id = f"prithivMLmods/siglip2-finetune"

try:
    api.create_repo(repo_id)
    print(f"Repo {repo_id} created")

except:

    print(f"Repo {repo_id} already exists")

api.upload_folder(
    folder_path="siglip2-finetune/",
    path_in_repo=".",
    repo_id=repo_id,
    repo_type="model",
    revision="main"
)
```

---

This complete script fine-tunes the SigLIP2 model for a single-label classification problem on deepfake image quality. Each section—from package installation to model upload—has been explained in detail to help you understand every step of the process.

## Computer Vision and Pattern Recognition

### Paper Reference

| Title | Link (Abstract) | Link (PDF) |
|--------|----------------|------------|
| SigLIP 2: Multilingual Vision-Language Encoders | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) | [PDF](https://arxiv.org/pdf/2502.14786) |

## Details and Benefits

SigLIP 2 is built on the foundation of Vision Transformers, ensuring backward compatibility with earlier versions. This allows users to replace model weights without overhauling their entire system. Unlike traditional contrastive loss, SigLIP 2 employs a sigmoid loss, enabling a more balanced learning of both global and local features.  

In addition to the sigmoid loss, SigLIP 2 integrates a decoder-based loss, enhancing tasks such as image captioning and region-specific localization. This leads to improved performance in dense prediction tasks. The model also incorporates a MAP head, which pools features from both image and text components, ensuring robust and detailed representations.  

A key innovation in SigLIP 2 is the *NaFlex variant*, which supports native aspect ratios by processing images at various resolutions using a single checkpoint. This approach preserves the spatial integrity of images, making it particularly effective for applications such as document understanding and OCR.  

Furthermore, self-distillation and masked prediction enhance the quality of local features. By training the model to predict masked patches, it learns to focus on subtle details critical for tasks like segmentation and depth estimation. This well-optimized design enables even smaller models to achieve superior performance through advanced distillation techniques.  

## Conclusion

SigLIP 2 represents a well-engineered and deliberate advancement in vision-language models. By integrating established techniques with thoughtful innovations, it effectively addresses key challenges such as fine-grained localization, dense prediction, and multilingual support. Moving beyond traditional contrastive losses, SigLIP 2 incorporates self-supervised objectives, leading to a more balanced and nuanced representation of visual data. Its careful handling of native aspect ratios through the NaFlex variant further enhances its applicability in real-world scenarios where preserving image integrity is crucial.  

The model's inclusion of multilingual data and de-biasing measures demonstrates an awareness of the diverse contexts in which it operates. This approach not only improves performance across various benchmarks but also aligns with broader ethical considerations in AI. Ultimately, the release of SigLIP 2 marks a significant step forward for the vision-language research community. It offers a versatile, backward-compatible framework that seamlessly integrates into existing systems. With its ability to deliver reliable performance across diverse tasks—while prioritizing fairness and inclusivity—SigLIP 2 sets a strong benchmark for future advancements in the field.  

Happy fine-tuning! 🤗
