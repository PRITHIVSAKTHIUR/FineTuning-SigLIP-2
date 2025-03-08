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
