# ShiftLoss: Decomposition-based Loss Function 개선

## 개선할 논문
뉴립스 2025
https://arxiv.org/pdf/2510.23672
https://github.com/decisionintelligence/DBLoss

## Introduction
<img width="803" height="204" alt="image" src="https://github.com/user-attachments/assets/3a78a35c-1206-4c02-b295-54759303954d" />

DBLoss는 시계열 데이터 후반부 예측을 위해 추세성과 계절성으로 손실함수를 분리해 계산하고 있다. 하지만 예측한 시계열 데이터를 시각화해보면 모양과 추세만 예측할 뿐, 시간 지연은 발생하고 있는 것을 확인할 수 있다. 그래서 나는 여기에 시간 정렬을 추가한다면 더욱 정확한 데이터 예측이 되지 않을까라는 가설을 세웠다.

## 알고리즘

손실함수에 추가 항을 넣는다. t = {-m, ..., -1, 0, 1, ..., m} 범위에서 가장 학습 loss가 낮은 것을 선택하여 학습한다. 
기존 논문에서 MSE/MAE는 시계열 데이터 예측의 평가 지표로 올바르지 않은 지표라 했다. 따라서, TDI와 DTW를 추가적으로 계산하여 비교한다.

# 현재 진행 상황

지금 전체 데이터를 한 번에 시간 정렬하다보니 오히려 TDI와 DTW가 감소하는 경우도 발생한다.
이를 해결하기 위해 윈도윙 함수로 변환할까 고민중이다.
하지만, 기존 논문 자체도 계산량 감소도 중요 포인트로 두었기에 윈도우 함수 추가는 오히려 계산량을 상승시키므로 적합한지에 대해 고민하게 된다.

## Contact

인하대학교 김재현(seankim0824@inha.edu)
