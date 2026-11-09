# ActivationLab — 116개 도감 정확성 점검 기록

## 점검 대상
- 총 **116개 항목**: 스칼라 98개, 게이트 11개, 벡터 7개.
- 기본·고전, ReLU·지수, 평활·현대, 주기·특수, 방사기저·커널, 수축·희소, 학습·적응, 벡터·확률, 게이트의 9개 계열.
- 도감 항목에는 동일 수식의 별칭(Elliott/Softsign, ReLU1/Hard Tanh, Soft Threshold/Softshrink), 매개변수형, 계산 근사형, 연구 예시가 포함된다.

## 발견·수정한 표현 오류
1. **도감 곡선과 세로축**: 출력값을 강제로 상수로 만들지 않는다. 작은 카드 및 상세 화면은 각 함수의 실제 평가값에 맞춰 y축을 자동 조절한다. 카드에 축 경계 숫자를 표시한다. 함수의 출력이 일정하지 않은데 그래프 오른쪽이 수평으로 끝나는 문제를 방지한다.
2. **불연속 함수**: Binary Step, Bipolar Step, Sign, Hardshrink, Thresholded ReLU의 점프를 단일 연결선으로 그리지 않는다. 열린 점·닫힌 점으로 경계 포함 여부를 구분한다. Thresholded ReLU의 θ=0은 점프가 아닌 꺾임으로 처리한다.
3. **LogSoftmax 미리보기**: 음수 로그 출력을 절댓값에 의한 양수 막대로 오해하지 않도록 0선 아래쪽으로 그린다. 본문의 출력값은 부호를 유지한다.
4. **게이트 구조**: GLU, ReGLU, GEGLU, SwiGLU, Bilinear GLU는 두 입력 (a,b)의 함수다. 카드의 단면은 b=1을 고정한 f(a,1)임을 명시한다.
5. **미분 가능성**: 기본 ELU(α=1), ELiSH의 1차 미분 가능성을 반영하고, ELU의 α가 바뀌면 상세 표시도 바뀌도록 한다. ReLU, Leaky ReLU 등의 꺾임 및 불연속점에서는 수치 도함수를 정의하지 않는다.
6. **도감 미리보기 애니메이션**: 곡선의 실제 SVG 경로를 따라 진행한다. 가시 영역의 카드만 하나의 requestAnimationFrame 루프로 갱신한다. 벡터 카드는 막대 채움으로 동작하고, 움직임 최소화 설정과 화면 전환·탭 비활성화를 존중한다.

## 수치·형태 검증
- 77개 스칼라·게이트 함수 기본 매개변수에서, 각 **x∈[−4,4]의 1,001개 표본**이 미리보기 세로축에 들어오는지 확인(총 77,077개 좌표).
- 116개 항목의 필수 필드·유형·매개변수 범위 확인. 스칼라·게이트 함수는 기본값 및 매개변수 끝값에서 비유한 출력이 없는지 검사.
- 독립적인 기준 수식과 경계값으로 Sigmoid, Tanh, SiLU, ReLU, Leaky ReLU, Softsign, Softplus, Hard Sigmoid, GELU 등 주요 함수 대조.
- 홀짝 대칭, 출력 범위, 실제 도함수(표본과 기준값), 단조성, 불연속 함수의 점프 개수와 열린 점·닫힌 점 개수를 확인.
- Softmax·Softmin·Sparsemax·Entmax 1.5의 비음수 출력·합 1, LogSoftmax의 exp(output) 합 1 및 큰 입력에서의 안정성을 검증.

## 시각화 해석 시 주의
- 작은 카드들은 각자의 **y축 범위가 다르므로** 기울기나 출력의 절대적 크기를 카드 사이에서 시각적으로 직접 비교하면 안 된다. 공통 스케일 비교는 비교 화면에서 한다.
- 비교 화면은 모두 같은 x: −6~6, y: −5~5 좌표계를 사용한다. 출력이 표시 영역을 넘어가면 **화면에서만 잘리며 수학적 함수값을 변형하지 않는다**.
- 벡터형은 확률 단체 등 벡터 전체의 연산이고, 게이트형은 두 입력의 상호작용을 갖는다. 하나의 f(x) 그래프로 전체 행위를 나타낼 수 없다.
- Hard Sigmoid는 PyTorch Hardsigmoid 정의(x/6+1/2, ±3에서 포화)를 사용한다. 프레임워크마다 다른 정의가 있을 수 있다.
- GELU의 정규분포 누적함수는 JS의 erf 근사로 계산한다. 수치근사는 분석값과 작은 오차가 있을 수 있다.
- '미분 가능'은 **1차 미분 가능**을 말하며 모든 차수에서 무한히 매끄럽다는 뜻이 아니다.
- 모든 고전·연구·파생 함수의 원저자 구현을 개별 실행 비교한 것은 아니다. 표본 기반 검사로 표본 사이의 모든 오류가 배제되는 것은 아니다.

## 참고자료
- [PyTorch — Activation Functions](https://docs.pytorch.org/docs/main/nn.html)
- [PyTorch — Hardsigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Hardsigmoid.html)
- [PyTorch — Hardshrink](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
- [PyTorch — GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU)
- [entmax 공식 구현](https://github.com/deep-spin/entmax)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

## 신규 확장 검증 (2026-11-09)
- 기존 항목 ID 001~082 유지, 추가 34개는 083~116.
- 방사기저·커널 계열 추가 및 기존 Gaussian·Morlet-like의 계열 재분류.
- RReLU는 학습 중 무작위 기울기가 아니라 추론 시 평균값만 사용. Maxout은 1차원 2가지 선형 분기의 단면.
- Hardmax는 첫 번째 동점 최댓값을 선택. Squared ReLU Normalize는 정규화 분모가 0일 때 균등 분포로 지정한 학습용 변형.
- 새로운 수식·매개변수·경계값 테스트를 추가하고, 각 카드의 간단한 설명은 상세 화면 제목 아래 기본 노출함.
- 원저자 코드와 모든 수치값을 일대일 재현하지는 않았으며 함수 정의와 샘플별 수치 검증으로 범위를 제한함.

출처: [PyTorch 활성화 함수](https://docs.pytorch.org/docs/main/nn.html), [Maxout Networks](https://proceedings.mlr.press/v28/goodfellow13.pdf), [400개 활성화 함수 조사](https://arxiv.org/abs/2402.09092), [Hard Mish](https://github.com/digantamisra98/H-Mish).
