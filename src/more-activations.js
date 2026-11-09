// Additional curated formulas; existing activation IDs are preserved by appending only.
export function appendMore({add,P,sigmoid,softplus,hs,special}){
const basic="basic",relu="relu",smooth="smooth",periodic="periodic",radial="radial",adaptive="adaptive",shrink="shrink";
add("Logistic β",basic,"f(x)=sigmoid(βx)",(x,p)=>sigmoid(p.beta*x),{desc:"Sigmoid의 전환 기울기 β를 조절하는 매개변수형입니다.",smooth:true,monotonic:true,params:[P("beta","β",.2,5,.2,1)],variant:"학습용 변형"});
add("Softsign β",basic,"f(x)=x/(1+β|x|)",(x,p)=>x/(1+p.beta*Math.abs(x)),{desc:"Softsign의 포화 속도를 β로 바꾸는 매개변수형입니다.",smooth:true,monotonic:true,params:[P("beta","β",.2,4,.2,1)],variant:"학습용 변형"});
add("LogCosh",basic,"f(x)=log(cosh(x))",x=>Math.abs(x)+softplus(-2*Math.abs(x))-Math.LN2,{desc:"작은 입력에서는 이차 함수처럼, 큰 절댓값에서는 선형처럼 증가하는 짝함수입니다.",smooth:true,monotonic:false});
add("RReLU (evaluation)",relu,"f(x)=x (x≥0), ((l+u)/2)x (x<0); l=1/8,u=1/3",x=>x>=0?x:x*(1/8+1/3)/2,{desc:"RReLU의 추론 모드입니다. 학습 중 무작위인 음수 기울기를 평균값으로 고정합니다.",monotonic:true});
add("Threshold",relu,"f(x)=x (x>θ), v (x≤θ)",(x,p)=>x>p.threshold?x:p.value,{desc:"임계값 이하를 지정된 값 v로 대체하는 PyTorch Threshold 형태입니다.",discontinuous:true,params:[P("threshold","임계 θ",0,3,.25,1),P("value","대체 v",-2,2,.25,-1)]});
add("MPELU",relu,"f(x)=x (x≥0), α(exp(βx)−1) (x<0)",(x,p)=>x>=0?x:p.alpha*Math.expm1(p.beta*x),{desc:"ELU의 음수 포화와 곡률을 따로 조절합니다. αβ=1일 때 원점에서 미분 가능합니다.",smooth:true,monotonic:true,trainable:true,params:[P("alpha","α",.2,2,.2,1),P("beta","β",.2,2,.2,1)]});
add("Centered Softplus",relu,"f(x)=softplus(x)−ln(2)",x=>softplus(x)-Math.LN2,{desc:"Softplus를 이동해 원점에서 0을 통과하게 한 학습용 변형입니다.",smooth:true,monotonic:true,variant:"학습용 변형"});
add("Soft Exponential (α>0)",smooth,"f(x)=(exp(αx)−1)/α+α (α>0)",(x,p)=>Math.expm1(p.alpha*x)/p.alpha+p.alpha,{desc:"Soft Exponential 계열에서 α가 양수인 구간만 시각화합니다.",smooth:true,monotonic:true,params:[P("alpha","α",.1,1,.1,.3)]});
add("Hard Mish",smooth,"f(x)=x·clamp((x+2)/2,0,1)",x=>x*Math.max(0,Math.min(1,(x+2)/2)),{desc:"Mish를 간단한 조각 다항식으로 근사한 Hard Mish입니다.",monotonic:false});
add("Cauchy RBF",radial,"f(x)=1/(1+(x/σ)²)",(x,p)=>1/(1+(x/p.sigma)**2),{desc:"꼬리가 비교적 완만하게 감소하는 코시 방사기저 함수입니다.",smooth:true,monotonic:false,params:[P("sigma","폭 σ",.4,3,.2,1)]});
add("Laplacian RBF",radial,"f(x)=exp(−|x|/σ)",(x,p)=>Math.exp(-Math.abs(x)/p.sigma),{desc:"원점에서 뾰족한 정점을 가지는 라플라스 방사기저입니다.",monotonic:false,params:[P("sigma","폭 σ",.4,3,.2,1)]});
add("Mexican Hat",radial,"f(x)=(1−(x/σ)²)exp(−x²/(2σ²))",(x,p)=>{const t=x/p.sigma;return (1-t*t)*Math.exp(-t*t/2);},{desc:"중앙의 양수 봉우리와 양옆의 음수 골을 갖는 웨이블릿입니다.",smooth:true,monotonic:false,params:[P("sigma","폭 σ",.5,3,.25,1)]});
add("Gabor",radial,"f(x)=cos(ωx)exp(−x²/(2σ²))",(x,p)=>Math.cos(p.omega*x)*Math.exp(-x*x/(2*p.sigma*p.sigma)),{desc:"국소적인 진동을 보여주는 가우시안 변조 코사인 함수입니다.",smooth:true,monotonic:false,params:[P("omega","각주파수 ω",.5,6,.5,3),P("sigma","폭 σ",.5,3,.25,1.5)]});
add("Gaussian σ",radial,"f(x)=exp(−x²/(2σ²))",(x,p)=>Math.exp(-x*x/(2*p.sigma*p.sigma)),{desc:"폭 σ를 직접 바꿀 수 있는 가우시안 방사기저 함수입니다.",smooth:true,monotonic:false,params:[P("sigma","폭 σ",.4,3,.2,1)],variant:"매개변수형"});
add("Super-Gaussian",radial,"f(x)=exp(−(x/σ)^(2p))",(x,p)=>Math.exp(-Math.pow(x/p.sigma,2*p.power)),{desc:"중심이 평평하고 바깥에서 급히 감소하는 고차 방사기저 함수입니다.",smooth:true,monotonic:false,params:[P("sigma","폭 σ",.5,3,.25,1),P("power","지수 p",2,4,1,2)]});
add("Sine ω",periodic,"f(x)=sin(ωx)",(x,p)=>Math.sin(p.omega*x),{desc:"주파수 ω를 조절해 사인 함수의 진동 횟수를 바꿉니다.",smooth:true,monotonic:false,params:[P("omega","ω",.5,8,.5,2)],variant:"매개변수형"});
add("Cosine ω",periodic,"f(x)=cos(ωx)",(x,p)=>Math.cos(p.omega*x),{desc:"주파수 ω를 조절하는 코사인 함수의 매개변수형입니다.",smooth:true,monotonic:false,params:[P("omega","ω",.5,8,.5,2)],variant:"매개변수형"});
add("Sinc ω",periodic,"f(x)=sin(ωx)/(ωx), f(0)=1",(x,p)=>Math.abs(x)<1e-9?1:Math.sin(p.omega*x)/(p.omega*x),{desc:"Sinc의 영점 간격을 주파수 ω로 조절합니다.",smooth:true,monotonic:false,params:[P("omega","ω",.5,8,.5,2)],variant:"매개변수형"});
add("Linear Sine",periodic,"f(x)=x·sin(ωx)",(x,p)=>x*Math.sin(p.omega*x),{desc:"입력 크기에 비례해 진폭이 커지는 학습용 합성 함수입니다.",smooth:true,monotonic:false,params:[P("omega","ω",.5,6,.5,1)],variant:"학습용 합성형"});
add("Maxout (2 branches)",adaptive,"f(x)=max(a₁x+b₁,a₂x+b₂)",(x,p)=>Math.max(p.a1*x+p.b1,p.a2*x+p.b2),{desc:"두 선형 가지 중 더 큰 값을 고르는 Maxout의 1차원 단면입니다.",trainable:true,params:[P("a1","a₁",-2,2,.2,1),P("b1","b₁",-2,2,.2,0),P("a2","a₂",-2,2,.2,-.2),P("b2","b₂",-2,2,.2,-.5)]});
add("Smooth Maxout",adaptive,"f(x)=log(exp(βu)+exp(βv))/β; u=a₁x+b₁,v=a₂x+b₂",(x,p)=>{const u=p.a1*x+p.b1,v=p.a2*x+p.b2,m=Math.max(u,v);return m+Math.log(Math.exp(p.beta*(u-m))+Math.exp(p.beta*(v-m)))/p.beta;},{desc:"두 선형 가지의 최댓값을 log-sum-exp로 매끄럽게 근사합니다.",smooth:true,trainable:true,params:[P("a1","a₁",-2,2,.2,1),P("b1","b₁",-2,2,.2,0),P("a2","a₂",-2,2,.2,-.2),P("b2","b₂",-2,2,.2,-.5),P("beta","β",.5,5,.5,2)],variant:"학습용 근사"});
add("APL (2 hinges)",adaptive,"f(x)=ReLU(x)+a₁ReLU(b₁−x)+a₂ReLU(b₂−x)",(x,p)=>Math.max(0,x)+p.a1*Math.max(0,p.b1-x)+p.a2*Math.max(0,p.b2-x),{desc:"두 개의 적응형 힌지로 조각별 선형 그래프를 학습하는 APL 예시입니다.",trainable:true,params:[P("a1","a₁",-1,1,.1,.5),P("b1","b₁",-3,3,.25,-1),P("a2","a₂",-1,1,.1,-.3),P("b2","b₂",-3,3,.25,1)]});
add("Parametric Arctan",adaptive,"f(x)=a·atan(bx)",(x,p)=>p.a*Math.atan(p.b*x),{desc:"아크탄젠트의 출력 크기와 입력 스케일을 조절합니다.",smooth:true,monotonic:true,params:[P("a","진폭 a",.2,3,.2,1),P("b","스케일 b",.2,3,.2,1)],variant:"학습용 매개변수형"});
add("Parametric Gaussian",radial,"f(x)=a·exp(−((x−μ)/σ)²)",(x,p)=>p.a*Math.exp(-Math.pow((x-p.mu)/p.sigma,2)),{desc:"진폭·중심·폭을 모두 조절할 수 있는 가우시안 방사기저 함수입니다.",smooth:true,monotonic:false,trainable:true,params:[P("a","진폭 a",.2,3,.2,1),P("mu","중심 μ",-2,2,.25,0),P("sigma","폭 σ",.4,3,.2,1)]});
add("Firmshrink",shrink,"f(x)=0 (|x|≤λ₁), sign(x)λ₂(|x|−λ₁)/(λ₂−λ₁) (λ₁<|x|<λ₂), x (|x|≥λ₂)",(x,p)=>{const v=Math.abs(x);return v<=p.low?0:v>=p.high?x:Math.sign(x)*p.high*(v-p.low)/(p.high-p.low);},{desc:"두 임계값 사이에서 출력을 연속적으로 복원하는 수축 연산입니다.",monotonic:true,params:[P("low","λ₁",.2,1.2,.1,.5),P("high","λ₂",1.5,4,.25,2)]});
add("Power Softshrink",shrink,"f(x)=sign(x)max(|x|−λ,0)^p",(x,p)=>Math.sign(x)*Math.pow(Math.max(Math.abs(x)-p.lambda,0),p.power),{desc:"Softshrink의 비영 영역을 거듭제곱해 작은 신호를 더 억제합니다.",smooth:true,monotonic:true,params:[P("lambda","임계 λ",.2,2,.2,.5),P("power","지수 p",2,4,1,2)],variant:"학습용 변형"});
const gate=()=>[P("gate","게이트 b",-4,4,.2,1)];
add("HardGLU","gated","f(a,b)=a·hard_sigmoid(b)",(x,p)=>x*hs(p.gate),{kind:"gated",desc:"Hard Sigmoid로 게이트를 만드는 GLU 변형입니다. 카드는 b=1 단면입니다.",params:gate()});
add("TanhGLU","gated","f(a,b)=a·tanh(b)",(x,p)=>x*Math.tanh(p.gate),{kind:"gated",desc:"Tanh로 양수·음수 게이트를 만듭니다. 카드는 b=1 단면입니다.",params:gate()});
add("SoftsignGLU","gated","f(a,b)=a·b/(1+|b|)",(x,p)=>x*p.gate/(1+Math.abs(p.gate)),{kind:"gated",desc:"Softsign을 게이트로 사용합니다. 카드는 b=1 단면입니다.",params:gate()});
add("ELUGLU","gated","f(a,b)=a·ELU(b)",(x,p)=>x*(p.gate>=0?p.gate:Math.expm1(p.gate)),{kind:"gated",desc:"ELU를 게이트에 적용합니다. 카드는 b=1 단면입니다.",params:gate()});
add("LiGLU","gated","f(a,b)=a·(b·tanh(b))",(x,p)=>x*p.gate*Math.tanh(p.gate),{kind:"gated",desc:"LiSHT를 게이트로 사용합니다. 카드는 b=1 단면입니다.",params:gate()});
add("SnakeGLU","gated","f(a,b)=a·(b+sin²(b))",(x,p)=>x*(p.gate+Math.sin(p.gate)**2),{kind:"gated",desc:"Snake(α=1)를 게이트로 사용합니다. 카드는 b=1 단면입니다.",params:gate()});
special("Hardmax","vector","pᵢ=1 (i=argmaxⱼzⱼ), 0 (그 외)","vector","가장 큰 로짓만 1로 남깁니다. 동점이라면 첫 번째 최댓값을 선택합니다.",null);
special("Squared ReLU Normalize","vector","pᵢ=ReLU(zᵢ)²/ΣⱼReLU(zⱼ)² (합=0이면 1/n)","vector","양수 로짓을 제곱해 정규화하는 학습용 변형입니다. 모두 0이면 균등 분포를 반환합니다.",null);
}
