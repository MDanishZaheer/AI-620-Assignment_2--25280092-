import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Default runtime settings for standalone execution.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

# This block defines and validates the test file path used for inference.
DATA_DIR = "DL_PA2/DL_PA2/processed_data"
TEST_FILE = os.path.join(DATA_DIR, "quickdraw_test.npz")

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Could not find test file: {TEST_FILE}")

# Champion model architecture used for inference.
class Champion(nn.Module):
    def __init__(self, input_dim=784, num_classes=15):
        super(Champion, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, num_classes)
        self.act_hidden = nn.GELU()
        self.drop = nn.Dropout(0.01)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        x = self.drop(self.act_hidden(self.bn1(self.fc1(x))))
        x = self.drop(self.act_hidden(self.bn2(self.fc2(x))))
        x = self.drop(self.act_hidden(self.bn3(self.fc3(x))))
        return self.out(x)

champ_model = Champion().to(DEVICE)

# This block loads the saved Champion checkpoint file.
CHAMP_FILE = "champ.pt" if os.path.exists("champ.pt") else "champ_best.pt"
if not os.path.exists(CHAMP_FILE):
    raise FileNotFoundError("Could not find model file: champ.pt or champ_best.pt")

champ_model.load_state_dict(torch.load(CHAMP_FILE, map_location=DEVICE))
champ_model.to(DEVICE)
champ_model.eval()

# This cell defines the custom QuickDrawDataset class.
# It supports both train mode (x,y) and test mode (x only).
# Tensor conversion and scaling are handled inside this class.
class QuickDrawDataset(Dataset):
    def __init__(self, file_path, mode='train'):
        self.mode = mode

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        print(f"Loading {mode} data from {file_path}...")
        data = np.load(file_path)

        if mode == 'train':
            self.x = data['x_train']
            self.y = data['y_train']
            self.classes = data['class_names']
            print(f"Loaded {len(self.x)} training samples. Classes: {len(self.classes)}")

        elif mode == 'test':
            self.x = data['test_images']
            self.y = None
            print(f"Loaded {len(self.x)} test images.")

        self.x = torch.from_numpy(self.x).float() / 255.0
        if self.y is not None:
            self.y = torch.from_numpy(self.y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.mode == 'train':
            label = self.y[idx]
            return img, label
        else:
            return img

# In this step we run test inference for leaderboard submission.
# Predicted labels are collected and converted to comma-separated format.
# Final submission.txt is generated for portal upload.
print("\n" + "="*40)
print("   GENERATING SUBMISSION FILE")
print("="*40)

test_dataset = QuickDrawDataset(TEST_FILE, mode='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_predictions(model, loader):
    model.eval()
    model.to(DEVICE)
    preds = []
    with torch.no_grad():
        for batch in loader:
            X = batch.to(DEVICE)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    return preds

print("Running inference on test set...")
predictions = get_predictions(champ_model, test_loader)

submission_file = "submission.txt"
print(f"Saving predictions to '{submission_file}'...")

submission_string = ",".join(map(str, predictions))

with open(submission_file, "w") as f:
    f.write(submission_string)
print(f"-> Copy & paste the results of this file to the portal.")
