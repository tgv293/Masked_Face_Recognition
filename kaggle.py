# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-18T04:23:45.553904Z","iopub.execute_input":"2025-03-18T04:23:45.554163Z","iopub.status.idle":"2025-03-18T04:24:03.097605Z","shell.execute_reply.started":"2025-03-18T04:23:45.554141Z","shell.execute_reply":"2025-03-18T04:24:03.096513Z"}}
from test import evalulate, test
from train import train2
from util import get_Optimizer2, get_Scheduler
from arcface import ArcFaceLoss
from torch.utils.data import DataLoader
from dataset import customized_dataset
from model import FaceNet2
import multiprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
import os
import torch
import condacolab
!pip install - q condacolab
condacolab.install()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-18T04:24:20.389093Z","iopub.execute_input":"2025-03-18T04:24:20.389474Z","iopub.status.idle":"2025-03-18T04:32:17.401518Z","shell.execute_reply.started":"2025-03-18T04:24:20.389440Z","shell.execute_reply":"2025-03-18T04:32:17.400315Z"}}
!conda install - c conda-forge cudatoolkit = 11.2 cudnn = 8.1.0
!conda install pytorch == 2.2.2 torchvision == 0.17.2 torchaudio == 2.2.2 pytorch-cuda = 11.8 - c pytorch - c nvidia - y

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-18T04:32:17.402881Z","iopub.execute_input":"2025-03-18T04:32:17.403207Z","iopub.status.idle":"2025-03-18T04:33:52.987752Z","shell.execute_reply.started":"2025-03-18T04:32:17.403183Z","shell.execute_reply":"2025-03-18T04:33:52.986945Z"}}
!pip install tensorflow == 2.10.1 numpy == 1.26.4 keras == 2.10
!pip install retina-face opencv-python pyyaml h5py
!pip install tensorflow-io
!pip install - U albumentations
# Install required libraries
!pip install facenet-pytorch
!pip install efficientnet-pytorch
!pip install timm
!pip install tqdm
!pip install scikit-learn matplotlib
!pip install pandas

# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:33:52.989713Z","iopub.execute_input":"2025-03-18T04:33:52.989989Z","iopub.status.idle":"2025-03-18T04:33:54.757833Z","shell.execute_reply.started":"2025-03-18T04:33:52.989967Z","shell.execute_reply":"2025-03-18T04:33:54.757024Z"}}
# Verify CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")


# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:33:54.759220Z","iopub.execute_input":"2025-03-18T04:33:54.759677Z","iopub.status.idle":"2025-03-18T04:33:54.981402Z","shell.execute_reply.started":"2025-03-18T04:33:54.759651Z","shell.execute_reply":"2025-03-18T04:33:54.980665Z"},"jupyter":{"outputs_hidden":false}}
# Check that the dataset is available
!ls / kaggle/input/casia-webmaskedface/CASIA-WebMaskedFace

# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:33:54.982298Z","iopub.execute_input":"2025-03-18T04:33:54.982569Z","iopub.status.idle":"2025-03-18T04:33:56.321513Z","shell.execute_reply.started":"2025-03-18T04:33:54.982546Z","shell.execute_reply":"2025-03-18T04:33:56.320561Z"},"jupyter":{"outputs_hidden":false}}
# Remove any previous clones to start fresh
!rm - rf Masked_Face_Recognition

# Clone the repository (fixed syntax - no spaces)
!git clone https: // github.com/SamYuen101234/Masked_Face_Recognition.git

# Create necessary directories
!mkdir - p Masked_Face_Recognition/models
!mkdir - p Masked_Face_Recognition/result
!mkdir - p Masked_Face_Recognition/Data

# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:33:56.322459Z","iopub.execute_input":"2025-03-18T04:33:56.322780Z","iopub.status.idle":"2025-03-18T04:33:57.226526Z","shell.execute_reply.started":"2025-03-18T04:33:56.322755Z","shell.execute_reply":"2025-03-18T04:33:57.225615Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:33:57.227607Z","iopub.execute_input":"2025-03-18T04:33:57.228098Z","iopub.status.idle":"2025-03-18T04:49:34.222615Z","shell.execute_reply.started":"2025-03-18T04:33:57.228062Z","shell.execute_reply":"2025-03-18T04:49:34.221743Z"},"jupyter":{"outputs_hidden":false}}
# Path to the dataset
data_path = "/kaggle/input/casia-webmaskedface/CASIA-WebMaskedFace"

# List all files and assign labels
files = []
targets = []
person_id = 0
person_dict = {}

print("Scanning dataset directory...")
# Walk through all directories
for root, dirs, filenames in os.walk(data_path):
    for filename in filenames:
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(root, filename)
            person_name = os.path.basename(os.path.dirname(full_path))

            if person_name not in person_dict:
                person_dict[person_name] = person_id
                person_id += 1

            files.append(full_path)
            targets.append(person_dict[person_name])

print(f"Found {len(files)} images from {len(person_dict)} unique identities")
# Create DataFrame
df = pd.DataFrame({'path': files, 'target': targets})

# Split into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['target'], random_state=42)
# Split the train set into train and validation sets (90% train, 10% validation)
train_df, valid_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df['target'], random_state=42)

# Save CSVs
train_df.to_csv('Masked_Face_Recognition/Data/train.csv', index=False)
valid_df.to_csv('Masked_Face_Recognition/Data/valid.csv', index=False)

print("Creating evaluation pairs...")
# Create evaluation sets (same and different pairs)
# For same-person pairs
same_pairs = []
for person in train_df['target'].unique():
    person_images = train_df[train_df['target'] == person]['path'].tolist()
    if len(person_images) >= 2:
        for i in range(min(len(person_images), 10)):  # Limit to 10 pairs per person
            img1 = person_images[i]
            img2 = person_images[(i+1) % len(person_images)]
            same_pairs.append({
                'path': img1,
                'target': person,
                'pair_path': img2,
                'pair_target': person
            })

# For different-person pairs
diff_pairs = []
all_targets = train_df['target'].unique()
for i in range(len(same_pairs)):
    person1 = same_pairs[i]['target']
    different_persons = [p for p in all_targets if p != person1]
    person2 = np.random.choice(different_persons)
    img1 = same_pairs[i]['path']
    img2 = train_df[train_df['target'] == person2]['path'].iloc[0]
    diff_pairs.append({
        'path': img1,
        'target': person1,
        'pair_path': img2,
        'pair_target': person2
    })

# Create evaluation DataFrames
eval_same_df = pd.DataFrame(same_pairs)
eval_diff_df = pd.DataFrame(diff_pairs)

# Save evaluation CSVs
eval_same_df.to_csv('Masked_Face_Recognition/Data/eval_same.csv', index=False)
eval_diff_df.to_csv('Masked_Face_Recognition/Data/eval_diff.csv', index=False)

print("Creating test pairs...")
# Create test pairs similar to eval pairs
test_same_pairs = []
test_diff_pairs = []

for person in test_df['target'].unique():
    person_images = test_df[test_df['target'] == person]['path'].tolist()
    if len(person_images) >= 2:
        for i in range(min(len(person_images), 5)):
            img1 = person_images[i]
            img2 = person_images[(i+1) % len(person_images)]
            test_same_pairs.append({
                'path': img1,
                'target': person,
                'pair_path': img2,
                'pair_target': person
            })

# Create different person pairs for test
for i in range(len(test_same_pairs)):
    person1 = test_same_pairs[i]['target']
    different_persons = [p for p in test_df['target'].unique() if p != person1]
    person2 = np.random.choice(different_persons)
    img1 = test_same_pairs[i]['path']
    img2 = test_df[test_df['target'] == person2]['path'].iloc[0]
    test_diff_pairs.append({
        'path': img1,
        'target': person1,
        'pair_path': img2,
        'pair_target': person2
    })

# Combine same and different pairs
test_pairs = test_same_pairs + test_diff_pairs
test_pairs_df = pd.DataFrame(test_pairs)
test_pairs_df.to_csv('Masked_Face_Recognition/Data/test.csv', index=False)

print(f"Created training set with {len(train_df)} images")
print(f"Created validation set with {len(valid_df)} images")
print(
    f"Created evaluation sets with {len(eval_same_df)} same pairs and {len(eval_diff_df)} different pairs")
print(f"Created test set with {len(test_pairs_df)} pairs")

# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:49:34.224297Z","iopub.execute_input":"2025-03-18T04:49:34.224561Z","iopub.status.idle":"2025-03-18T04:49:34.230639Z","shell.execute_reply.started":"2025-03-18T04:49:34.224538Z","shell.execute_reply":"2025-03-18T04:49:34.229794Z"},"jupyter":{"outputs_hidden":false}}

% % writefile train_model1.py

# Change directory to the repository root
os.chdir('Masked_Face_Recognition')

# Add repository to Python path for imports
sys.path.insert(0, os.path.abspath('.'))


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataframes
df_train = pd.read_csv('Data/train.csv')
df_valid = pd.read_csv('Data/valid.csv')
df_eval1 = pd.read_csv('Data/eval_same.csv')
df_eval2 = pd.read_csv('Data/eval_diff.csv')
df_test = pd.read_csv('Data/test.csv')

# Config parameters
BATCH_SIZE = 64
NUM_WORKERS = 2
embedding_size = 512
num_classes = df_train.target.nunique()
weight_decay = 5e-4
lr = 1e-1
dropout = 0.4
model_name = None  # Use InceptionResNetV1
pretrain = True
loss_fn = 'arcface'
pool = None  # Use default pooling
scheduler_name = 'multistep'
optimizer_type = 'sgd'
num_epochs = 25
eval_every = 100
arcface_s = 45
arcface_m = 0.4
class_weights_norm = 'batch'
crit = "focal"
name = 'arcface_model1.pth'

print(f"Training with {num_classes} classes")

# Create datasets and dataloaders
train_dataset = customized_dataset(df_train, mode='train')
valid_dataset = customized_dataset(df_valid, mode='valid')
eval_dataset1 = customized_dataset(df_eval1, mode='eval')
eval_dataset2 = customized_dataset(df_eval2, mode='eval')
test_dataset = customized_dataset(df_test, mode='test')

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
eval_loader1 = DataLoader(
    eval_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
eval_loader2 = DataLoader(
    eval_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=NUM_WORKERS)

# Calculate class weights for arcface loss
val_counts = df_train.target.value_counts().sort_index().values
class_weights = 1/np.log1p(val_counts)
class_weights = (class_weights / class_weights.sum()) * num_classes
# Ensure it's on the right device
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

# Initialize arcface loss
metric_crit = ArcFaceLoss(arcface_s, arcface_m, crit, weight=class_weights,
                          class_weights_norm=class_weights_norm).to(device)

# Initialize model
print("Initializing model...")
facenet = FaceNet2(num_classes=num_classes,
                   model_name=model_name,
                   pool=pool,
                   embedding_size=embedding_size,
                   dropout=dropout,
                   device=device,
                   pretrain=pretrain).to(device)

# Initialize optimizer and scheduler
optimizer = get_Optimizer2(
    facenet, metric_crit, optimizer_type, lr, weight_decay)
scheduler = get_Scheduler(optimizer, lr, scheduler_name)

print("Model initialized successfully")

try:
    # Train the model
    print("Starting training...")
    train2(facenet, train_loader, eval_loader1, eval_loader2, metric_crit, optimizer, scheduler,
           num_epochs, eval_every, num_classes, device, name)

    # Evaluate the model
    dist_threshold = evalulate(
        facenet, eval_loader1, eval_loader2, device, loss_fn)
    print(f'Distance threshold: {dist_threshold}')

    # Test the model
    test(facenet, test_loader, dist_threshold, device, loss_fn)

    print('Training and evaluation complete!')
except Exception as e:
    import traceback
    print(f"Error during execution: {e}")
    traceback.print_exc()


# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T04:49:34.231637Z","iopub.execute_input":"2025-03-18T04:49:34.231909Z","iopub.status.idle":"2025-03-18T05:16:31.080596Z","shell.execute_reply.started":"2025-03-18T04:49:34.231880Z","shell.execute_reply":"2025-03-18T05:16:31.079774Z"},"jupyter":{"outputs_hidden":false}}
# Execute the script
!chmod + x train_model1.py
!python train_model1.py


# %% [code] {"execution":{"iopub.status.busy":"2025-03-18T05:16:31.081557Z","iopub.execute_input":"2025-03-18T05:16:31.081800Z","iopub.status.idle":"2025-03-18T05:16:31.222257Z","shell.execute_reply.started":"2025-03-18T05:16:31.081776Z","shell.execute_reply":"2025-03-18T05:16:31.221463Z"},"jupyter":{"outputs_hidden":false}}
# Archive results
!zip - r model1_arcface.zip Masked_Face_Recognition/models/arcface_model1.pth Masked_Face_Recognition/result/
