import {appendMore} from "./more-activations.js";
// ActivationLab v1.0 — 검증 가능한 스칼라·벡터·게이트 함수 카탈로그
export const families = [
  {id:"basic",label:"기본·고전",color:"#64748b"},
  {id:"relu",label:"ReLU·지수",color:"#3b82f6"},
  {id:"smooth",label:"평활·현대",color:"#8b5cf6"},
  {id:"periodic",label:"주기·특수",color:"#e59c30"},
  {id:"radial",label:"방사기저·커널",color:"#168a9b"},
  {id:"shrink",label:"수축·희소",color:"#14a88b"},
  {id:"adaptive",label:"학습·적응",color:"#e46b91"},
  {id:"vector",label:"벡터·확률",color:"#0ca6bc"},
  {id:"gated",label:"게이트",color:"#af74e4"}
];
export const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
export const softplus=x=>x>35?x:x< -35?Math.exp(x):Math.log1p(Math.exp(x));
const erf=x=>{const s=Math.sign(x),a=Math.abs(x),t=1/(1+.3275911*a);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-a*a));};
const gelu=x=>x*(1+erf(x/Math.SQRT2))/2;
const geluApprox=x=>x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)))/2;
const hs=x=>Math.max(0,Math.min(1,(x+3)/6));
const P=(key,label,min,max,step,value)=>({key,label,min,max,step,value});
const alpha=(value=.1)=>P("alpha","α",0,1,.05,value);
export const activations=[];
function add(name,family,formula,fn,opts={}){const id=String(activations.length+1).padStart(3,"0");activations.push({id,name,family,formula,fn,kind:"scalar",smooth:false,monotonic:null,params:[],desc:"함수의 출력과 기울기를 직접 살펴보세요.",...opts});}
const basic="basic",relu="relu",smooth="smooth",periodic="periodic",shrink="shrink",adaptive="adaptive";
add("Identity",basic,"f(x) = x",x=>x,{desc:"입력값을 그대로 반환하는 선형 함수입니다.",smooth:true,monotonic:true,alias:"Linear"});
add("Binary Step",basic,"f(x) = 1 (x ≥ 0), 0 (x < 0)",x=>x>=0?1:0,{desc:"임계값에서 출력이 바뀌는 계단 함수입니다. 그 지점에서 미분되지 않습니다.",monotonic:true,discontinuous:true,alias:"Heaviside"});
add("Bipolar Step",basic,"f(x) = 1 (x ≥ 0), −1 (x < 0)",x=>x>=0?1:-1,{desc:"두 출력값 −1, 1만 갖는 이극성 계단 함수입니다.",monotonic:true,discontinuous:true});
add("Sign",basic,"f(x) = sign(x)",x=>Math.sign(x),{desc:"입력의 부호를 −1, 0, 1로 반환합니다.",monotonic:true,discontinuous:true,alias:"Signum"});
add("Sigmoid",basic,"f(x) = 1 / (1 + exp(−x))",x=>sigmoid(x),{desc:"출력 범위가 (0,1)인 로지스틱 함수입니다.",smooth:true,monotonic:true,alias:"Logistic"});
add("Bipolar Sigmoid",basic,"f(x) = 2·sigmoid(x) − 1",x=>2*sigmoid(x)-1,{desc:"시그모이드를 (−1,1) 범위로 변형한 함수입니다.",smooth:true,monotonic:true});
add("Tanh",basic,"f(x) = tanh(x)",x=>Math.tanh(x),{desc:"원점 대칭이며 출력 범위가 (−1,1)입니다.",smooth:true,monotonic:true,alias:"Hyperbolic Tangent"});
add("Hard Sigmoid",basic,"f(x) = clamp((x + 3)/6, 0, 1)",x=>hs(x),{desc:"PyTorch Hardsigmoid 정의(x/6+1/2)를 사용합니다. 라이브러리에 따라 다른 Hard Sigmoid 정의도 있습니다.",monotonic:true,alias:"Hardsigmoid"});
add("Hard Tanh",basic,"f(x) = clamp(x, −1, 1)",x=>Math.max(-1,Math.min(1,x)),{desc:"Tanh를 절단된 선형 함수로 근사합니다.",monotonic:true,alias:"Hardtanh"});
add("Softsign",basic,"f(x) = x / (1 + |x|)",x=>x/(1+Math.abs(x)),{desc:"양끝에서 점진적으로 포화되는 유계 함수입니다.",smooth:true,monotonic:true});
add("Arctan",basic,"f(x) = atan(x)",x=>Math.atan(x),{desc:"아크탄젠트 기반의 완만한 포화 함수입니다.",smooth:true,monotonic:true});
add("LogSigmoid",basic,"f(x) = log(sigmoid(x)) = −softplus(−x)",x=>-softplus(-x),{desc:"확률의 로그를 안정적으로 계산하는 데 활용됩니다.",smooth:true,monotonic:true});
add("Elliott",basic,"f(x) = x / (1 + |x|)",x=>x/(1+Math.abs(x)),{desc:"Elliott의 원형은 Softsign과 동일한 수식입니다. 별칭 관계를 명시했습니다.",smooth:true,monotonic:true,alias:"Softsign의 별칭"});
add("Bent Identity",basic,"f(x) = x + (√(x²+1) − 1)/2",x=>x+(Math.sqrt(x*x+1)-1)/2,{desc:"항등함수에 완만한 비선형 항을 더합니다.",smooth:true,monotonic:true});
add("Absolute",basic,"f(x) = |x|",x=>Math.abs(x),{desc:"입력의 크기만 남겨 부호 정보를 제거합니다.",monotonic:false,alias:"Abs"});
add("Softplus",relu,"f(x) = log(1 + exp(x))",x=>softplus(x),{desc:"ReLU의 매끄러운 근사로 사용됩니다.",smooth:true,monotonic:true});
add("ReLU",relu,"f(x) = max(0, x)",x=>Math.max(0,x),{desc:"음수 출력을 0으로 만들고 양수를 그대로 통과시킵니다.",monotonic:true,alias:"Rectified Linear Unit"});
add("Leaky ReLU",relu,"f(x) = max(x, αx)",(x,p)=>x>=0?x:p.alpha*x,{desc:"음수 구간에도 작은 기울기를 남깁니다.",monotonic:true,params:[alpha(.1)]});
add("PReLU",relu,"f(x) = max(x, αx), α는 학습 대상",(x,p)=>x>=0?x:p.alpha*x,{desc:"음수 구간 기울기 α를 학습하는 변형입니다. 여기서는 수동 조절만 제공합니다.",monotonic:true,trainable:true,params:[alpha(.25)]});
add("ReLU6",relu,"f(x) = min(max(0, x), 6)",x=>Math.min(6,Math.max(0,x)),{desc:"출력을 [0,6]으로 제한합니다.",monotonic:true});
add("ReLU1",relu,"f(x) = clamp(x, −1, 1)",x=>Math.max(-1,Math.min(1,x)),{desc:"문헌과 프레임워크에서 ReLU1은 Hard Tanh와 동일하게 정의될 수 있습니다.",monotonic:true,alias:"Hard Tanh의 별칭"});
add("Squared ReLU",relu,"f(x) = max(0, x)²",x=>Math.max(0,x)**2,{desc:"ReLU의 양수 출력을 제곱합니다.",monotonic:true,alias:"ReLU²"});
add("Cubed ReLU",relu,"f(x) = max(0, x)³",x=>Math.max(0,x)**3,{desc:"양의 구간에서 세제곱 비선형성을 갖습니다.",smooth:true,monotonic:true,alias:"ReLU³"});
add("Thresholded ReLU",relu,"f(x) = x (x > θ), 0 (그 외)",(x,p)=>x>p.theta?x:0,{desc:"설정한 임계값 이하의 출력을 제거하며, θ>0에서는 x=θ에서 불연속입니다.",monotonic:true,params:[P("theta","θ",0,3,.25,1)],discontinuous:true});
add("Shifted ReLU",relu,"f(x) = max(0, x − θ)",(x,p)=>Math.max(0,x-p.theta),{desc:"ReLU의 꺾이는 위치를 오른쪽으로 이동합니다.",monotonic:true,params:[P("theta","θ",-2,2,.25,1)]});
add("Clipped ReLU",relu,"f(x) = clamp(x, 0, c)",(x,p)=>Math.max(0,Math.min(p.cap,x)),{desc:"양수 출력의 최대 크기를 제한합니다.",monotonic:true,params:[P("cap","상한 c",1,6,.5,3)]});
add("ELU",relu,"f(x) = x (x≥0), α(exp(x)−1) (x<0)",(x,p)=>x>=0?x:p.alpha*Math.expm1(x),{desc:"음수 영역을 지수 곡선으로 연결합니다. α=1일 때 미분 가능합니다.",params:[P("alpha","α",.1,2,.1,1)],smooth:true,monotonic:true});
add("CELU",relu,"f(x) = x (x≥0), α(exp(x/α)−1) (x<0)",(x,p)=>x>=0?x:p.alpha*Math.expm1(x/p.alpha),{desc:"ELU의 연속 미분 성질을 보강한 변형입니다(α>0).",smooth:true,monotonic:true,params:[P("alpha","α",.2,2,.1,1)]});
add("SELU",relu,"f(x) = λ·[x (x≥0), α(exp(x)−1) (x<0)]",x=>1.0507009873554805*(x>=0?x:1.6732632423543772*Math.expm1(x)),{desc:"특정 초기화·네트워크 조건에서 자기 정규화 특성을 유도합니다.",monotonic:true});
add("ISRLU",relu,"f(x) = x (x≥0), x/√(1+αx²) (x<0)",(x,p)=>x>=0?x:x/Math.sqrt(1+p.alpha*x*x),{desc:"역제곱근 기반의 ReLU 변형입니다.",smooth:true,monotonic:true,params:[P("alpha","α",.1,3,.1,1)]});
add("Softplus β",relu,"f(x) = log(1 + exp(βx)) / β",(x,p)=>softplus(p.beta*x)/p.beta,{desc:"β가 커질수록 ReLU에 가까워지는 평활 근사입니다.",smooth:true,monotonic:true,params:[P("beta","β",.2,5,.2,1)]});
add("Scaled Softplus",relu,"f(x) = α·log(1+exp(x))",(x,p)=>p.alpha*softplus(x),{desc:"Softplus 출력의 스케일을 조절합니다.",smooth:true,monotonic:true,params:[P("alpha","α",.2,2,.1,1)]});
add("Parametric ELU",adaptive,"f(x) = (a/b)x (x≥0), a(exp(x/b)−1) (x<0)",(x,p)=>x>=0?p.a*x/p.b:p.a*Math.expm1(x/p.b),{desc:"양수 기울기와 음수 포화 정도를 학습 가능한 매개변수로 조절합니다.",smooth:true,monotonic:true,trainable:true,params:[P("a","a",.2,2,.1,1),P("b","b",.2,2,.1,1)],alias:"PELU"});
add("SReLU",adaptive,"f(x) = tₗ+aₗ(x−tₗ) (x≤tₗ); x (tₗ<x<tᵣ); tᵣ+aᵣ(x−tᵣ) (x≥tᵣ)",(x,p)=>x<=p.tl?p.tl+p.al*(x-p.tl):x>=p.tr?p.tr+p.ar*(x-p.tr):x,{desc:"양쪽 임계값과 기울기를 학습하는 S자형 ReLU입니다.",trainable:true,monotonic:true,params:[P("tl","왼쪽 임계 tₗ",-3,0,.25,-1),P("tr","오른쪽 임계 tᵣ",0,3,.25,1),P("al","왼쪽 기울기",0,2,.1,.2),P("ar","오른쪽 기울기",0,2,.1,.2)]});
add("APL (1 hinge)",adaptive,"f(x) = max(0,x) + a·max(0,b−x)",(x,p)=>Math.max(0,x)+p.a*Math.max(0,p.b-x),{desc:"적응형 조각 선형 함수(APL)의 단일 힌지 예시입니다.",trainable:true,params:[P("a","a",-1,1,.1,.3),P("b","b",-2,2,.25,0)]});
add("ACON-C",adaptive,"f(x) = (p₁−p₂)x·sigmoid(β(p₁−p₂)x)+p₂x",(x,p)=>(p.p1-p.p2)*x*sigmoid(p.beta*(p.p1-p.p2)*x)+p.p2*x,{desc:"학습 가능한 계수로 활성화 강도를 조절하는 형태입니다.",smooth:true,trainable:true,params:[P("p1","p₁",0,2,.1,1),P("p2","p₂",-1,1,.1,0),P("beta","β",.2,4,.2,1)]});
add("Parametric Tanh",adaptive,"f(x) = a·tanh(bx)",(x,p)=>p.a*Math.tanh(p.b*x),{desc:"Tanh의 출력 크기와 기울기를 매개변수화한 형태입니다.",smooth:true,monotonic:true,params:[P("a","a",.2,2,.1,1),P("b","b",.2,3,.1,1)]});
add("Scaled SiLU",adaptive,"f(x) = a·x·sigmoid(bx)",(x,p)=>p.a*x*sigmoid(p.b*x),{desc:"SiLU의 크기와 입력 기울기를 독립적으로 조절합니다.",smooth:true,monotonic:false,params:[P("a","a",.2,2,.1,1),P("b","b",.2,3,.1,1)]});
add("GELU",smooth,"f(x) = x·Φ(x)",x=>gelu(x),{desc:"표준정규분포 누적분포함수 Φ로 정의됩니다. 브라우저 계산에서는 erf 수치 근사를 사용합니다.",smooth:true,monotonic:false,alias:"Gaussian Error Linear Unit"});
add("GELU (tanh)",smooth,"f(x) ≈ x/2·[1+tanh(√(2/π)(x+0.044715x³))]",x=>geluApprox(x),{desc:"GELU에서 널리 쓰이는 tanh 근사입니다.",smooth:true,monotonic:false});
add("QuickGELU",smooth,"f(x) = x·sigmoid(1.702x)",x=>x*sigmoid(1.702*x),{desc:"빠른 계산을 위한 GELU 근사형입니다.",smooth:true,monotonic:false});
add("SiLU",smooth,"f(x) = x·sigmoid(x)",x=>x*sigmoid(x),{desc:"부드러운 비단조 함수로 Swish(β=1)와 동일합니다.",smooth:true,monotonic:false,alias:"Swish"});
add("Swish β",smooth,"f(x) = x·sigmoid(βx)",(x,p)=>x*sigmoid(p.beta*x),{desc:"β로 곡선의 굽힘을 조절합니다.",smooth:true,monotonic:false,params:[P("beta","β",.2,4,.2,1)]});
add("E-Swish",smooth,"f(x) = βx·sigmoid(x)",(x,p)=>p.beta*x*sigmoid(x),{desc:"SiLU 출력의 배율을 조정하는 변형입니다.",smooth:true,monotonic:false,params:[P("beta","β",.2,3,.1,1.5)]});
add("Hard Swish",smooth,"f(x) = x·clamp((x+3)/6,0,1)",x=>x*hs(x),{desc:"SiLU의 계산 비용을 낮춘 조각 선형 기반 근사형입니다.",monotonic:false,alias:"Hardswish"});
add("Mish",smooth,"f(x) = x·tanh(softplus(x))",x=>x*Math.tanh(softplus(x)),{desc:"Softplus와 Tanh를 결합한 평활 비단조 함수입니다.",smooth:true,monotonic:false});
add("Squareplus",smooth,"f(x) = (x+√(x²+b))/2",(x,p)=>(x+Math.sqrt(x*x+p.b))/2,{desc:"제곱근 연산을 사용하는 ReLU 평활 근사입니다.",smooth:true,monotonic:true,params:[P("b","b",.25,9,.25,4)]});
add("LiSHT",smooth,"f(x) = x·tanh(x)",x=>x*Math.tanh(x),{desc:"양·음의 큰 입력 모두 양의 방향으로 증가하는 함수입니다.",smooth:true,monotonic:false});
add("dSiLU",smooth,"f(x) = sigmoid(x) + x·sigmoid(x)(1−sigmoid(x))",x=>{const s=sigmoid(x);return s+x*s*(1-s)},{desc:"SiLU의 1차 도함수를 활성화 함수로 사용하는 형태입니다.",smooth:true,monotonic:false});
add("Logish",smooth,"f(x) = x·log(1+sigmoid(x))",x=>x*Math.log1p(sigmoid(x)),{desc:"로그와 시그모이드의 조합을 사용합니다.",smooth:true,monotonic:false});
add("TanhExp",smooth,"f(x) = x·tanh(exp(x))",x=>x*Math.tanh(Math.exp(Math.min(x,50))),{desc:"양의 방향에서 선형에 가까워지는 평활 함수입니다.",smooth:true,monotonic:false});
add("ELiSH",smooth,"f(x) = xσ(x) (x≥0), (exp(x)−1)σ(x) (x<0)",x=>x>=0?x*sigmoid(x):Math.expm1(x)*sigmoid(x),{desc:"ELU와 Sigmoid를 조합한 조각별 함수입니다. 0에서 1차 도함수가 연속입니다.",smooth:true,monotonic:false});
add("Hard ELiSH",smooth,"f(x) = x·hard_sigmoid(x) (x≥0), (exp(x)−1)·hard_sigmoid(x) (x<0)",x=>x>=0?x*hs(x):Math.expm1(x)*hs(x),{desc:"ELiSH의 게이트를 Hard Sigmoid로 대체한 변형입니다.",monotonic:false});
add("Exponential",smooth,"f(x) = exp(x)",x=>Math.exp(x),{desc:"입력을 양수 영역에 사상하는 지수 함수입니다.",smooth:true,monotonic:true});
add("Sine",periodic,"f(x) = sin(x)",x=>Math.sin(x),{desc:"주기성을 갖는 연속 함수입니다.",smooth:true,monotonic:false,alias:"Sin"});
add("Cosine",periodic,"f(x) = cos(x)",x=>Math.cos(x),{desc:"사인 함수에서 위상을 이동한 주기 함수입니다.",smooth:true,monotonic:false,alias:"Cos"});
add("Sinc",periodic,"f(x) = sin(x)/x, f(0)=1",x=>Math.abs(x)<1e-9?1:Math.sin(x)/x,{desc:"정의는 sin(x)/x(비정규화 Sinc)이며, x=0에서는 극한값 1로 연장합니다.",smooth:true,monotonic:false});
add("Snake",periodic,"f(x) = x + sin²(αx)/α",(x,p)=>x+Math.sin(p.alpha*x)**2/p.alpha,{desc:"주기적 성분을 선형 추세 위에 결합한 함수입니다. 도함수는 1+sin(2αx)로 항상 음수가 아닙니다.",smooth:true,monotonic:true,params:[P("alpha","α",.2,4,.2,1)]});
add("SnakeBeta",periodic,"f(x) = x + sin²(αx)/β",(x,p)=>x+Math.sin(p.alpha*x)**2/p.beta,{desc:"주기 성분의 주파수 α와 진폭 β를 독립적으로 조절합니다. α/β에 따라 단조성이 달라집니다.",smooth:true,monotonic:null,params:[P("alpha","α",.2,4,.2,1),P("beta","β",.2,4,.2,1)]});
add("Gaussian","radial","f(x) = exp(−x²)",x=>Math.exp(-x*x),{desc:"0을 중심으로 하는 종 모양의 방사 기저형 함수입니다.",smooth:true,monotonic:false,alias:"Gaussian RBF"});
add("GCU",periodic,"f(x) = x·cos(x)",x=>x*Math.cos(x),{desc:"진동하는 주기 성분과 입력값의 곱입니다.",smooth:true,monotonic:false,alias:"Growing Cosine Unit"});
add("Morlet-like","radial","f(x) = cos(5x)·exp(−x²/2)",x=>Math.cos(5*x)*Math.exp(-x*x/2),{desc:"정규화·평균 보정을 생략한 Morlet 형태의 예시입니다.",smooth:true,monotonic:false});
add("Sin²",periodic,"f(x) = sin²(x)",x=>Math.sin(x)**2,{desc:"음수 출력이 없고 π 주기를 갖는 함수입니다.",smooth:true,monotonic:false});
add("Cos²",periodic,"f(x) = cos²(x)",x=>Math.cos(x)**2,{desc:"0과 1 사이에서 진동합니다.",smooth:true,monotonic:false});
add("Hardshrink",shrink,"f(x) = x (|x|>λ), 0 (그 외)",(x,p)=>Math.abs(x)>p.lambda?x:0,{desc:"작은 절대값의 입력을 0으로 제거합니다.",discontinuous:true,params:[P("lambda","λ",.1,2,.1,.5)],monotonic:true});
add("Softshrink",shrink,"f(x) = x−λ (x>λ), x+λ (x<−λ), 0 (그 외)",(x,p)=>x>p.lambda?x-p.lambda:x< -p.lambda?x+p.lambda:0,{desc:"임계값 이내는 0으로 만들고 바깥 구간은 선형 이동합니다. 경계 ±λ에서는 미분되지 않습니다.",monotonic:true,params:[P("lambda","λ",.1,2,.1,.5)]});
add("Tanhshrink",shrink,"f(x) = x − tanh(x)",x=>x-Math.tanh(x),{desc:"입력에서 Tanh 출력을 뺀 함수입니다.",smooth:true,monotonic:true});
add("Softclip",shrink,"f(x) = softplus(x) − softplus(x−c)",(x,p)=>softplus(x)-softplus(x-p.c),{desc:"매끄럽게 0과 상한 c 사이로 포화되는 함수입니다.",smooth:true,monotonic:true,params:[P("c","상한 c",1,6,.5,3)]});
add("Triangle",shrink,"f(x) = max(0,1−|x|)",x=>Math.max(0,1-Math.abs(x)),{desc:"중심에서 가장 크고 양쪽에서 0이 되는 삼각 함수형 활성화입니다.",monotonic:false});
add("Bump",shrink,"f(x) = exp(−1/(1−x²)) (|x|<1), 0 (그 외)",x=>Math.abs(x)<1?Math.exp(-1/(1-x*x)):0,{desc:"−1과 1 밖에서는 정확히 0이며 모든 차수에서 미분 가능합니다.",smooth:true,monotonic:false});
add("Soft Threshold",shrink,"f(x) = sign(x)·max(|x|−λ,0)",(x,p)=>Math.sign(x)*Math.max(0,Math.abs(x)-p.lambda),{desc:"Softshrink와 동치인 소프트 임계 연산입니다.",monotonic:true,params:[P("lambda","λ",.1,2,.1,.5)],alias:"Softshrink의 별칭"});
add("Gaussian Derivative",shrink,"f(x) = −2x·exp(−x²)",x=>-2*x*Math.exp(-x*x),{desc:"가우시안 함수의 1차 도함수입니다.",smooth:true,monotonic:false});
function special(name,family,formula,kind,desc,fn){add(name,family,formula,fn,{kind,desc,smooth:false});}
special("Softmax","vector","pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)","vector","로짓 벡터를 합이 1인 확률 분포로 변환합니다.",null);
special("LogSoftmax","vector","yᵢ = zᵢ − log Σⱼ exp(zⱼ)","vector","Softmax 출력을 로그 공간에서 안정적으로 계산합니다.",null);
special("Softmin","vector","pᵢ = exp(−zᵢ) / Σⱼ exp(−zⱼ)","vector","낮은 입력값에 높은 확률을 부여합니다.",null);
special("Sparsemax","vector","pᵢ = max(zᵢ − τ,0), Σᵢpᵢ=1","vector","일부 확률을 정확히 0으로 만드는 확률 단체 투영입니다.",null);
special("Entmax 1.5","vector","pᵢ = max(zᵢ/2 − τ,0)², Σᵢpᵢ=1","vector","Softmax와 Sparsemax 사이의 희소한 확률 분포를 만듭니다.",null);
special("GLU","gated","f(a,b) = a·sigmoid(b)","gated","별도 입력 b가 a의 흐름을 제어합니다.",(x,p)=>x*sigmoid(p.gate));
special("ReGLU","gated","f(a,b) = a·ReLU(b)","gated","ReLU로 게이트 크기를 조절합니다.",(x,p)=>x*Math.max(0,p.gate));
special("GEGLU","gated","f(a,b) = a·GELU(b)","gated","GELU 기반 게이팅을 사용합니다.",(x,p)=>x*gelu(p.gate));
special("SwiGLU","gated","f(a,b) = a·SiLU(b)","gated","SiLU로 게이트를 조절하는 Transformer 관련 변형입니다.",(x,p)=>x*p.gate*sigmoid(p.gate));
special("Bilinear GLU","gated","f(a,b) = a·b","gated","비선형 게이트 없이 두 입력의 곱을 사용합니다.",(x,p)=>x*p.gate);
for(const a of activations.filter(a=>a.kind==="gated")){a.params=[P("gate","게이트 입력 b",-4,4,.2,1)];a.monotonic=null;}
appendMore({add,P,sigmoid,softplus,hs,special});
export function defaults(a){return Object.fromEntries(a.params.map(p=>[p.key,p.value]));}
export function evaluate(a,x,params=defaults(a)){if(!a.fn)return NaN;const n=a.fn(x,params);return Number.isFinite(n)?n:NaN;}
export function isDifferentiable(a,p=defaults(a)){if(a.kind!=="scalar")return null;if(a.name==="ELU")return Math.abs(p.alpha-1)<1e-9;if(a.name==="MPELU")return Math.abs(p.alpha*p.beta-1)<1e-9;if(a.name==="Leaky ReLU"||a.name==="PReLU")return Math.abs(p.alpha-1)<1e-9;if(a.name==="SReLU")return Math.abs(p.al-1)<1e-9&&Math.abs(p.ar-1)<1e-9;return Boolean(a.smooth);}
export function derivative(a,x,params=defaults(a)){if(!a.fn)return NaN;const h=1e-4*Math.max(1,Math.abs(x)),left=evaluate(a,x-h,params),center=evaluate(a,x,params),right=evaluate(a,x+h,params);if(![left,center,right].every(Number.isFinite))return NaN;if(a.discontinuous&&Math.abs(right-left)>.02)return NaN;const ls=(center-left)/h,rs=(right-center)/h;if(!isDifferentiable(a,params)&&Math.abs(ls-rs)>.025*(1+Math.max(Math.abs(ls),Math.abs(rs))))return NaN;const value=(right-left)/(2*h);return Number.isFinite(value)?value:NaN;}
export function computeVector(kind,values){if(kind==="Softmax"||kind==="Softmin"||kind==="LogSoftmax"){const z=kind==="Softmin"?values.map(v=>-v):values;const m=Math.max(...z);const s=z.map(v=>Math.exp(v-m));const logsum=Math.log(s.reduce((a,b)=>a+b,0));return kind==="LogSoftmax"?z.map(v=>v-m-logsum):s.map(v=>v/s.reduce((a,b)=>a+b,0));}if(kind==="Hardmax"){const winner=values.indexOf(Math.max(...values));return values.map((_,i)=>i===winner?1:0);}if(kind==="Squared ReLU Normalize"){const weights=values.map(v=>Math.max(0,v)**2),total=weights.reduce((a,b)=>a+b,0);return total>0?weights.map(v=>v/total):values.map(()=>1/values.length);}if(kind==="Sparsemax"){const sorted=[...values].sort((a,b)=>b-a);let sum=0,k=0,tau=0;sorted.forEach((z,i)=>{sum+=z;if(1+(i+1)*z>sum){k=i+1;tau=(sum-1)/k;}});return values.map(v=>Math.max(0,v-tau));}if(kind==="Entmax 1.5"){const z=values.map(v=>v/2),hi=Math.max(...z),lo=Math.min(...z)-2;let left=lo,right=hi;for(let i=0;i<70;i++){const mid=(left+right)/2;const s=z.reduce((t,v)=>t+Math.max(0,v-mid)**2,0);if(s>1)left=mid;else right=mid;}const tau=(left+right)/2;return z.map(v=>Math.max(0,v-tau)**2);}return values;}
