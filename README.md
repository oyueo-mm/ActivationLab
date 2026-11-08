# ActivationLab

신경망 활성화 함수를 탐색하고 직접 비교할 수 있는 인터랙티브 도감입니다.

**[웹앱 열기](https://oyueo-mm.github.io/ActivationLab/)**

## 현재 기능

- **홈 화면**: ReLU / Sigmoid / Tanh / SiLU 네 가지를 실제 함수값 기준으로 그리며 애니메이션 표시 (x: −4~4, 서로 다른 y축 범위를 숫자로 명시)
- **82개 항목 모두 그래프 미리보기 애니메이션**: 화면에 보이는 카드만 그리기·점 추적(벡터형은 막대 순차 표시), 화면 전환 시 자동 정지, 모션 줄이기 설정 준수
- **82개 항목**: 8개 계열(기본·고전, ReLU·지수, 평활·현대, 주기·특수, 수축·희소, 학습·적응, 벡터·확률, 게이트)
- 한글 검색, 계열·유형·특성 필터링
- 개별 함수 그래프와 수치 미분 기반 도함수 오버레이
- 매개변수 슬라이더를 통한 실시간 조작
- 최대 4개 함수의 동일 좌표계 비교 및 속성 표
- Softmax / LogSoftmax / Softmin / Sparsemax / Entmax 1.5 벡터 실험
- GLU / ReGLU / GEGLU / SwiGLU / Bilinear GLU 게이트 단면 시각화
- 데스크톱·모바일 반응형 레이아웃, 키보드 단축키 `/`(검색), `Escape`(닫기)

### 정확성 관련 주의

- **82개는 도감 항목 수**로, 별칭(Softsign / Elliott, Hard Tanh / ReLU1, Softshrink / Soft Threshold)과 매개변수형·근사형을 포함합니다.
- 스칼라 함수의 도함수는 **수치 미분**이며, 꺾이는 지점에서의 그래프 값은 수학적 미분 가능성을 뜻하지 않습니다.
- 미분 불가능한 꺾임·불연속 지점에서 수치 도함수는 정의하지 않고, 그 밖의 구간에서는 계산할 수 있습니다.
- 불연속 함수의 그래프는 점프 구간을 연결하지 않습니다. 열린 점·닫힌 점을 경계에서 구분하여 표시합니다.
- 벡터·게이트 함수는 일반적인 단일 스칼라 함수와 다릅니다. 벡터는 성분별 출력(미리보기 기준 로짓 `[1.8, 0.2, -0.7, 1]`), 게이트는 두 번째 입력 `b=1`을 고정한 단면을 보여줍니다. **LogSoftmax 미리보기의 음수 막대는 0 아래쪽으로 표시**합니다.
- 학습 가능한 매개변수는 v1.0에서 **수동 조절만 가능**하며, 실제 신경망 학습 실험은 차기 버전 범위입니다.
- 도감 카드와 상세 그래프는 함수값에 따라 세로축 범위가 자동 조정됩니다(x: 카드 −4~4, 상세 −6~6). 비교 화면만 공통 좌표계 x∈[−6,6], y∈[−5,5]를 사용하여, 범위 밖 출력은 **함수값을 바꾸지 않고 화면에서 클리핑**합니다.
- 홈 그래프의 세로축 범위는 카드마다 다릅니다. 실제 출력값은 카드 아래 `x · y`로 확인할 수 있습니다.
- 수학적 정의가 프레임워크에 따라 다른 Hard Sigmoid는 **PyTorch** 버전을 사용합니다. GELU의 Φ 함수는 브라우저에서 `erf` 수치 근사로 계산합니다.
- **82개는 독립 원천 함수 수가 아니라 도감 항목 수**이며, 이름이 다른 동일 함수·파생형·근사형을 포함합니다.
- [정확성 점검 기록](./AUDIT.md): 테스트 범위, 알려진 제약과 수식·시각화 확인 항목.

## 로컬 실행

외부 API·서버·빌드 도구 없이 정적 파일로 동작합니다. ES 모듈을 사용하므로 로컬에서는 HTTP 서버로 열어야 합니다.

```bash
python3 -m http.server 8000
```

그다음 `http://localhost:8000`을 열면 됩니다. 테스트는 Node.js 22 이상에서 `npm test`로 수행합니다.

## 배포

`main`에 코드가 반영되면 `.github/workflows/pages.yml`이 테스트 후 GitHub Pages에 정적 파일을 배포합니다.

최초 한 번은 저장소 **Settings → Pages → Build and deployment → Source = GitHub Actions**로 설정해야 합니다.

## 참고

- [PyTorch torch.nn](https://docs.pytorch.org/docs/stable/nn.html)
- [Keras activations](https://keras.io/activations/)
- [GLU Variants Improve Transformer (arXiv)](https://arxiv.org/abs/2002.05202)
- [Gaussian Error Linear Units (arXiv)](https://arxiv.org/abs/1606.08415)

MIT License
