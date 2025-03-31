import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from MNIST.model import SimpleCNN
from CIFAR10.resnet import resnet20
from attacks import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted
import os

# 공격 이미지 저장 폴더 생성
os.makedirs('adv_images', exist_ok=True)

# 디바이스 설정 (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= MNIST 모델 로딩 및 데이터 준비 =========
mnist_model = SimpleCNN().to(device)
mnist_model.load_state_dict(torch.load("./MNIST/mnist_model_20.pth", map_location=device))
mnist_model.eval()

mnist_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True), 
    batch_size=1
)

# MNIST 이미지 1장 가져오기 (디바이스 이동)
x_mnist, y_mnist = next(iter(mnist_loader))
x_mnist, y_mnist = x_mnist.to(device), y_mnist.to(device)
target_mnist = torch.tensor([(y_mnist.item() + 1) % 10]).to(device) # 정답 클래스에서 +1만큼 이동한 클래스를 타겟으로 임의 설정

# MNIST에 대한 4가지 공격 수행
mnist_attacks = {
    'FGSM_targeted': fgsm_targeted(mnist_model, x_mnist, target_mnist, eps=0.3),
    'FGSM_untargeted': fgsm_untargeted(mnist_model, x_mnist, y_mnist, eps=0.3),
    'PGD_targeted': pgd_targeted(mnist_model, x_mnist, target_mnist, k=50, eps=0.3, eps_step=0.05),
    'PGD_untargeted': pgd_untargeted(mnist_model, x_mnist, y_mnist, k=50, eps=0.3, eps_step=0.05)
}

# MNIST 결과 출력 및 이미지 저장
for name, x_adv in mnist_attacks.items():
    orig_pred = mnist_model(x_mnist).argmax().item()
    adv_pred = mnist_model(x_adv).argmax().item()
    print(f"[MNIST-{name}] 원본 예측: {orig_pred}, 공격 후 예측: {adv_pred}")
    
    utils.save_image(x_mnist.cpu(), f'adv_images/MNIST_{name}_ori.png')
    utils.save_image(x_adv.cpu(), f'adv_images/MNIST_{name}_adv.png')

# ========= CIFAR-10 모델 로딩 및 데이터 준비 =========
cifar_model = resnet20().to(device)

# pretrained 가중치 로딩 및 'module.' 제거
checkpoint = torch.load("./CIFAR10/pretrained_models/resnet20-12fca82f.th", map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
cifar_model.load_state_dict(new_state_dict)
cifar_model.eval()

cifar_loader = DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True),
    batch_size=1
)

# CIFAR-10 이미지 1장 가져오기 (디바이스 이동)
x_cifar, y_cifar = next(iter(cifar_loader))
x_cifar, y_cifar = x_cifar.to(device), y_cifar.to(device)
target_cifar = torch.tensor([(y_cifar.item() + 1) % 10]).to(device) # 정답 클래스에서 +1만큼 이동한 클래스를 타겟으로 임의 설정

# CIFAR-10에 대한 4가지 공격 수행
cifar_attacks = {
    'FGSM_targeted': fgsm_targeted(cifar_model, x_cifar, target_cifar, eps=0.05),
    'FGSM_untargeted': fgsm_untargeted(cifar_model, x_cifar, y_cifar, eps=0.05),
    'PGD_targeted': pgd_targeted(cifar_model, x_cifar, target_cifar, k=50, eps=0.05, eps_step=0.01),
    'PGD_untargeted': pgd_untargeted(cifar_model, x_cifar, y_cifar, k=50, eps=0.05, eps_step=0.01)
}

# CIFAR-10 결과 출력 및 이미지 저장
for name, x_adv in cifar_attacks.items():
    orig_pred = cifar_model(x_cifar).argmax().item()
    adv_pred = cifar_model(x_adv).argmax().item()
    print(f"[CIFAR10-{name}] 원본 예측: {orig_pred}, 공격 후 예측: {adv_pred}")
    
    utils.save_image(x_cifar.cpu(), f'adv_images/CIFAR10_{name}_ori.png')
    utils.save_image(x_adv.cpu(), f'adv_images/CIFAR10_{name}_adv.png')

print("모든 공격 이미지가 adv_images 폴더에 저장되었습니다.")
