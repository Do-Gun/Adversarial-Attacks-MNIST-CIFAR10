# 🛡️ Adversarial Attacks on MNIST & CIFAR-10 (FGSM & PGD)

MNIST와 CIFAR-10 데이터셋에 대해 FGSM 및 PGD 기반의 Adversarial Attacks 구현

MNIST에 대하여 간단한 CNN 구조, CIFAR-10 Dataset에 대해서는 Pretrain-ResNet20 모델 사용

각 알고리즘마다 Test Datasets 중 한 장의 이미지를 불러와 공격을 수행
<br>
<br>

# 📁 폴더 구조
```Adversarial-Attacks-MNIST-CIFAR10/
├── MNIST/
│   │   ......
│   ├── model.py               # SimpleCNN 구조 정의
│   └── mnist_model_20.pth     # 학습된 MNIST 모델 가운치
│
├── CIFAR10/
│   │   ......
│   ├── resnet.py              # CIFAR용 ResNet 정의 
│   └── pretrained_models/
│       └── resnet20-12fca82f.th  # CIFAR-10에 대한 사전 학습된 가중치
│
├── attacks.py                # FGSM/PGD (targeted/untargeted) 공격 함수 정의
├── test.py                   # MNIST/CIFAR-10에 대한 공격 실행 스크립트
├── adv_images/               # 생성된 adversarial 이미지 저장 폴더, test.py 실행시 자동으로 설치
├── .gitignore                # 대용량 파일 제외
└── README.md
```
<br>

# 🎯 지원되는 공격 종류

FGSM (Targeted / Untargeted)

PGD (Targeted / Untargeted
<br>
<br>


# 🧪 테스트 환경 설정
**CUDA 12.4 + PyTorch 2.4.1** 환경 기준으로 다음과 같이 설정 (bash)

**1. 가상환경 생성**

```conda create -n rtai python=3.8```

```conda activate rtai```

**2. 필요 라이브러리 설치**

```pip install torch==2.4.1 torchvision==0.19.1```

**3. Git clone**
   
```git clone https://github.com/Do-Gun/Adversarial-Attacks-MNIST-CIFAR10```

```cd Adversarial-Attacks-MNIST-CIFAR10```

**4. 테스트 실행**

```python test.py```
<br>
<br>



# 📈 결과 예시 출력

[MNIST-FGSM_targeted] 원본 예측: 7, 공격 후 예측: 3

[MNIST-PGD_targeted] 원본 예측: 3, 공격 후 예측: 4

[CIFAR10-FGSM_untargeted] 원본 예측: 3, 공격 후 예측: 5

...

**원본 이미지와 공격 받은 이미지는 /Adversarial-Attacks-MNIST-CIFAR10/adv_images 경로에 저장됩니다.**

**타겟은 원본 + 1 라벨을 예측하도록 설정했습니다**
<br>
<br>




# 🧠 모델 출처
CIFAR-10 Pretrained ResNet20

본 프로젝트에서는 https://github.com/akamaster/pytorch_resnet_cifar10 구현을 기반으로 ResNet20 구조(resnet.py)를 가져와 사용하였으며, 해당 저장소에서 제공하는 resnet20-12fca82f.th 사전 학습된 가중치를 활용했습니다.
