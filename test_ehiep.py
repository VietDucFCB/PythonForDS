import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import classification_report
import numpy as np

# Định nghĩa dữ liệu mẫu (dùng dữ liệu Iris để minh họa)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, Y = iris.data, iris.target
x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.33, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


# Dataset và DataLoader
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


train_dataset = CustomDataset(x_train, y_train)
val_dataset = CustomDataset(x_val, y_val)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Định nghĩa mô hình
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Khởi tạo mô hình, hàm mất mát, và bộ tối ưu
model = SimpleNet(input_dim=4, hidden_dim=25, output_dim=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Khởi tạo các metric
accuracy_metric = Accuracy(task="multiclass", num_classes=3).to(device)
precision_metric = Precision(task="multiclass", num_classes=3, average="macro").to(device)
recall_metric = Recall(task="multiclass", num_classes=3, average="macro").to(device)
f1_metric = F1Score(task="multiclass", num_classes=3, average="macro").to(device)


# Hàm huấn luyện
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_train_preds.extend(outputs.argmax(dim=1).cpu())
            all_train_labels.extend(y_batch.cpu())

        # Tính toán các chỉ số trên tập huấn luyện
        all_train_preds = torch.tensor(all_train_preds)
        all_train_labels = torch.tensor(all_train_labels)
        train_accuracy = accuracy_metric(all_train_preds, all_train_labels)
        train_precision = precision_metric(all_train_preds, all_train_labels)
        train_recall = recall_metric(all_train_preds, all_train_labels)
        train_f1 = f1_metric(all_train_preds, all_train_labels)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, "
              f"Train Recall: {train_recall:.4f}, "
              f"Train F1: {train_f1:.4f}")

        # Đặt lại các metric
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu())
                val_labels.extend(y_batch.cpu())

        # Tính toán các chỉ số trên tập validation
        val_preds = torch.tensor(val_preds)
        val_labels = torch.tensor(val_labels)
        val_accuracy = accuracy_metric(val_preds, val_labels)
        val_precision = precision_metric(val_preds, val_labels)
        val_recall = recall_metric(val_preds, val_labels)
        val_f1 = f1_metric(val_preds, val_labels)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Validation Precision: {val_precision:.4f}, "
              f"Validation Recall: {val_recall:.4f}, "
              f"Validation F1: {val_f1:.4f}")

        # Reset các metric sau mỗi epoch
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()


# Huấn luyện mô hình
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


# Hàm suy luận và đánh giá trên tập test
def inference(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # In báo cáo phân loại
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=iris.target_names))


# Đánh giá mô hình trên tập test
inference(model, test_loader)