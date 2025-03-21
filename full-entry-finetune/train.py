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
from datasets import Dataset, Image, ClassLabel, load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator,
    AutoModel,
    AutoProcessor
)
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
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------
# Load and preprocess dataset
# -----------------------
dataset = load_dataset("--your--dataset--goes--here--", split="train")

# Build DataFrame from dataset (for oversampling)
file_names = []
labels = []
for example in dataset:
    file_path = str(example['image'])
    label = example['label']
    file_names.append(file_path)
    labels.append(label)

print(len(file_names), len(labels))
df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
print("DataFrame shape:", df.shape)
print(df.head())
print("Unique labels:", df['label'].unique())

# Oversample to balance classes
y = df[['label']]
df_no_label = df.drop(['label'], axis=1)
ros = RandomOverSampler(random_state=83)
df_resampled, y_resampled = ros.fit_resample(df_no_label, y)
df_resampled['label'] = y_resampled
df = df_resampled  # use the oversampled DataFrame
del y, y_resampled, df_no_label
gc.collect()

# Define label mappings (adjust labels_list as needed)
labels_list = ['example_label_1', 'example_label_2']
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}
ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
print("Mapping of IDs to Labels:", id2label)
print("Mapping of Labels to IDs:", label2id)

# Update dataset with label mapping
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)

# Use the full dataset for fine-tuning (no train-test split)
full_data = dataset

# -----------------------
# Define image processing and transformations
# -----------------------
from transformers import AutoImageProcessor, SiglipForImageClassification

model_str = "google/siglip2-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_str)

# Extract parameters from processor
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Define training and validation transforms
_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(90),
    RandomAdjustSharpness(2),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])
_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=image_mean, std=image_std)
])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Create training and evaluation datasets with different transforms
train_data = full_data.with_transform(train_transforms)
eval_data = full_data.with_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# -----------------------
# Load model and set configuration
# -----------------------
model = SiglipForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))
model.config.id2label = id2label
model.config.label2id = label2id

print("Trainable parameters (in millions):", model.num_parameters(only_trainable=True) / 1e6)

# -----------------------
# Define compute_metrics
# -----------------------
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy_metric.compute(predictions=predicted_labels, references=label_ids)['accuracy']
    return {"accuracy": acc_score}

# -----------------------
# Set up TrainingArguments and Trainer
# -----------------------
args = TrainingArguments(
    output_dir="siglip2-finetune-full",
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch on eval_data
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
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# -----------------------
# Fine-tuning: Evaluation, Training, and Prediction
# -----------------------
# Optional evaluation before training
trainer.evaluate()

# Fine-tune the model on the full dataset
trainer.train()

# Evaluate after training
trainer.evaluate()

# Get predictions and compute metrics
outputs = trainer.predict(eval_data)
print("Prediction metrics:", outputs.metrics)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

# -----------------------
# Plot confusion matrix and print classification report
# -----------------------
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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) <= 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

# -----------------------
# Save and upload the model
# -----------------------
trainer.save_model()

from huggingface_hub import notebook_login, HfApi
notebook_login()

api = HfApi()
repo_id = "prithivMLmods/siglip2-finetune-full"
try:
    api.create_repo(repo_id)
    print(f"Repo {repo_id} created")
except Exception as e:
    print(f"Repo {repo_id} already exists or could not be created: {e}")

api.upload_folder(
    folder_path="siglip2-finetune-full/",
    path_in_repo=".",
    repo_id=repo_id,
    repo_type="model",
    revision="main"
)
