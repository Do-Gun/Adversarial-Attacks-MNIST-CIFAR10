import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import SimpleCNN

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로딩
train_data = datasets.MNIST('../data', download=True, transform=transforms.ToTensor())
train_set, val_set = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

# 모델을 GPU로 이동
model = SimpleCNN().to(device)

# 학습 코드 (정확한 저장명으로 저장되도록 수정)
def train(model, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch}/{epochs}] Train:{train_loss:.4f}, Val:{val_loss:.4f}")

        # 10 에폭마다 정확한 이름으로 가중치 저장
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'mnist_model_{epoch}.pth')

if __name__ == "__main__":
    train(model)
