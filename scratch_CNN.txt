# CNN MODELS
-------------------------------------


## **LeNet-5**

**framework**:Tensorflow,Keras

**DATASET**:*CIFAR-10*

**학습이미지**:50,000장

**테스트이미지**:10,000장

신경망 학습의 정확도와 손실 그래프


<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/7227b41c-6adf-4ccb-87e4-db44a1dee1b8" />


궁금증


1.x_train,x_test를 왜 flaot하고 255.0으로 나누는가?


2.dropout를 하는 이유


궁금증에 대한 나의 정리된 생각


1.신경망 내부의 계산에서 실수값이 나올 수 있기에 flaot으로 바꾸어주고 255.0으로 나누는 이유는 train과test해야하는 data크기가 크기에 수월하게 학습시키고 기울기 발산을 방지하기 위해 255.0으로 나눠서 픽셀 값을 0~1로 두면 신경망이 계산 중 오버피팅을 방지 할 수 있고 학습이 더 빠르게 가능하다.


2.신경망 학습 도중 나오는 오버피팅을 방지하기 위한 기법이다


한계점


신경망 층이 얕음으로 대규모 학습에 불리하다.

-------------------------------------

## **VGGNet16**

**framework**:Tensorflow,Keras

**DATASET**:*cats_vs_dogs*

**학습이미지**:약18,400장

**테스트이미지**:약4,600장

신경망 학습의 정확도와 손실 그래프

<img width="1010" height="470" alt="download" src="https://github.com/user-attachments/assets/23975c50-ace8-4a7e-96ec-ec350d8a9f75" />

신경망 구현 도중 생긴 문제와 해결 방법

VGGnet16을 구현하기 위해 conv층과 fc층을 모두 구현하여 epochs을 50으로 하고 lr는1e-4로 두어 학습하려 하였으나 학습 시간이 너무 오래걸렸고 epochs15번째쯤에 정확도가 55%부근에서 머물러서 직접 con과fc를 구현하여 학습시키는 방법보다는 미리 학습된 라이브러리를 불러와 epochs을 3으로 줄이고 adam를 사용하여 lr를 0.001로 두어 학습 시켰다.

그럼에도 불구하고 학습속도가 그리 빠르지않아 prefetch를 사용하여 epochs이 끝나기전에 미리 data를 준비해주는 함수를 활용하여 학습속도를 향상시켰다.

VGGnet16의 한계점

단순한 대규모 data를 학습시키는 것은 가능하지만 복잡한 대규모 data에서는 약한 모습을 보인다.

많은 GPU를 사용할 수록 학습 속도나 대규모 data를 학습 할 수 있지만 효율성이 떨어진다.

-------------------------------------

## **GoogLeNet**

**framework**:Tensorflow,keras

**DATASET**:*cats_vs_dogs*

**학습이미지**:약18,400장

**테스트이미지**:약4,600장

신경망 학습의 정확도와 손실 그래프

<img width="1165" height="470" alt="download" src="https://github.com/user-attachments/assets/cebdcddc-523e-4844-8585-ce8eae8f3e43" />

신경망 구현 후 느낀점

GoogLenet를구현할때 vggnet보다 더 적은 파라미터를 이용하고 병렬 계산이다보니 고화질이나 정확도 측면에서 매우 좋은 결과를 보였다.하지만 필터를 1x1,3x3,5x5를 섞어서 사용하다보니 복잡성이 증가하여 코드 구현에서의 오류가 많이 발생할 수 있다 생각했다.

한계점

병렬 계산이다 보니 코드오류 발생시 수정에 어려움이 있을거라 생각했다.

GoogLenet에서부터는 vggnet보다 적은 파라미터로 더 깊은 신경망을 구현 할 수 있는데 깊은 신경망을 구현할 때 Vanishing Gradient문제가 발생 할 수 있다고 생각했다.

-------------------------------------

## **ResNet-50**

**framework**:Tensorflow,keras

**DATASET**:*cats_vs_dogs*

**학습이미지**:약18,400장

**테스트이미지**:약4,600장

신경망 학습의 정확도와 손실 그래프

<img width="1010" height="448" alt="download" src="https://github.com/user-attachments/assets/f28080ab-bb5e-4b81-b448-1785f362b7bd" />

신경망 구현 후 느낀점

GoogLenet에서 더욱 깊은 층을 구현할 수록 발생하는 Vanishing Gradient을 해결하고자 잔차 학습을 이용하여 나온 ResNet-50을 구현해봤다.

Dense(1, activation='sigmoid')를 이용하여 개와고양이를 맞춘 확률을 나타내었는데 정확도가 78%로 그리 높지는 않았다.

그 이유로는 그래프에서 보이는 것처럼 Vaildation Accuracy가 흔들리는데 overfitting가능성이 보인다고 생각했다. 또 한 bach_size를 처음에 32로 잡았다가 학습이 너무 느려 128로 잡았는데 64로 잡았으면 정확도가 조금 더 개선되었을거라 생각한다. 추가적으로 epochs를 충분히 늘렸다면 손실은 낮아지고 정확도는 더욱 높아졌을것이다.

top부분을 제외하고 dense로 출력층을 압축함으로 계산량을 줄이고 미세 조정을 통해 초기계층은 미리 학습된 가중치를 이용하고 뒷쪽 convolution을 고정된 가중치를 받아 계산함함으로써 계산량을 최소화하였다.

