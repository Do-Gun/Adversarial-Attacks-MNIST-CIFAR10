# Adversarial Attacks (MNIST & CIFAR-10)

🛡️ Adversarial Attacks on MNIST & CIFAR-10 (FGSM & PGD)
MNIST와 CIFAR-10 데이터셋에 대해 FGSM (Fast Gradient Sign Method) 및 PGD (Projected Gradient Descent) 기반의 adversarial 공격을 구현함.

MNIST에 대하여는 간단한 CNN 구조의 모델을 사용, CIFAR-10 데이터셋에 대해서는 Pretrain된 ResNet20 모델 사용함.


🎯 지원되는 공격 종류
FGSM (Targeted / Untargeted)
PGD (Targeted / Untargeted


🧪 테스트 환경 설정
CUDA 12.4 + PyTorch 2.4.1 환경 기준으로 다음과 같이 설정합니다.

# 1. 가상환경 생성
conda create -n rtai python=3.8
conda activate rtai

# 2. 필요 라이브러리 설치
pip install torch==2.4.1 torchvision==0.19.1

# 3. Git clone
git clone https://github.com/Do-Gun/Adversarial-Attacks-MNIST-CIFAR10
cd Adversarial-Attacks-MNIST-CIFAR10

# 4. 테스트 실행
python test.py


📈 결과 예시 출력
[MNIST-FGSM_targeted] 원본 예측: 7, 공격 후 예측: 3
[MNIST-PGD_targeted] 원본 예측: 7, 공격 후 예측: 2
[CIFAR10-FGSM_untargeted] 원본 예측: 3, 공격 후 예측: 5
...
원본 이미지와 공격 받은 이미지는 /Adversarial-Attacks-MNIST-CIFAR10/adv_images 경로에 저장됩니다.


📁 폴더 구조
Adversarial-Attacks-MNIST-CIFAR10/
├── MNIST/
│   ├── model.py               # SimpleCNN 구조 정의
│   └── mnist_model_20.pth     # 학습된 MNIST 모델 가운치
│
├── CIFAR10/
│   ├── resnet.py              # ResNet20 CIFAR용 정의 (akamaster)
│   │   ......
│   └── pretrained_models/
│       └── resnet20-12fca82f.th  # CIFAR-10에 대한 사전 학습된 가중치
│
├── attacks.py                # FGSM/PGD (targeted/untargeted) 공격 함수 정의
├── test.py                   # MNIST/CIFAR-10에 대한 공격 실행 스크립트
├── adv_images/               # 생성된 adversarial 이미지 저장 폴더
├── .gitignore                # 대용량 파일 제외
└── README.md                 # 프리젝트 설명


🧠 모델 출처
CIFAR-10 Pretrained ResNet20출처: akamaster/pytorch_resnet_cifar10해당 레포지토리에서 제공하는 resnet20-12fca82f.th 모델을 사용했습니다.
