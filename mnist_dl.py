import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import onnx

# デバイス設定（GPUが利用可能ならGPUを使用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# CNNモデルの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7x64 -> 128
        self.fc2 = nn.Linear(128, 10)  # 128 -> 10 (分類クラス数)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # フラット化
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_mnist():
    # データセットのロード（MNIST）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    # モデルのインスタンス化とデバイスへの転送
    model = CNN().to(device)
    
    # 損失関数と最適化アルゴリズム
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    epochs = 3
    start_time = time.time()

    for epoch in range(epochs):
        model.train()  # モデルを訓練モードに
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 勾配をゼロにリセット
            outputs = model(inputs)  # 順伝播
            loss = criterion(outputs, labels)  # 損失計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータ更新

            running_loss += loss.item()

            if i % 100 == 99:  # 100ミニバッチごとに出力
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} completed")
        

    # 学習後のモデル評価
    model.eval()  # モデルを評価モードに
    correct = 0
    total = 0

    with torch.no_grad():  # 評価時は勾配計算を無効化
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # 学習時間の計測
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    # モデルの保存（.ptファイルとして）
    torch.save(model.state_dict(), "cnn_mnist_model.pt")
    print("Model saved as .pt file")

    model.load_state_dict(torch.load("results/cnn_mnist_model.pt", weights_only=True))
    model.eval()  # 推論モードに設定
    torch.load("results/cnn_mnist_model.pt")


def onnx():
    # モデルのインスタンス化とデバイスへの転送
    model = CNN().to(device)

    # 学習後のモデル評価
    model.eval()

    model.load_state_dict(torch.load("results/cnn_mnist_model.pt", weights_only=True))
    model.eval()  # 推論モードに設定
    torch.load("results/cnn_mnist_model.pt")

    # ONNX形式で保存
    # ダミー入力テンソル（MNISTの場合は28x28の画像）
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # ONNXファイルとして保存
    torch.onnx.export(model, dummy_input, "results/cnn_mnist_model.onnx", verbose=True, opset_version=15)
    print("Model saved as .onnx file")


if __name__ == "__main__":
    # train or onnx
    mode = "train"
    mode = "onnx"

    if mode == "train":
        train_mnist()
        onnx()
    else:
        onnx()
