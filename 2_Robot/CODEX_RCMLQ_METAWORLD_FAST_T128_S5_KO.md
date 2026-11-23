# Codex 실행 명세: RC-MLQ × Meta-World 고속 실험

**최종 고정 설정: d=4, T=128, 5개 독립 interaction, 6개 알고리즘, 7개 (M, beta) 조합.**

## 0. 너에게 맡기는 작업

현재 작업 디렉터리의 기존 코드를 확인하고, 아래 실험을 **구현 → 검증 → 속도 측정 → 실제 실행 → 그림·표·보고서 생성**까지 완료하라. 계획이나 TODO 코드만 작성하고 끝내지 마라. 실행 가능한 로컬 환경과 권한이 있으면 실제 실험을 진행하라.

목표는 **결과의 정확성과 실험 설정을 유지하면서 전체 실행시간을 최소화하는 것**이다. CPU와 GPU를 모두 활용하되, 단순히 사용률을 100%로 만드는 것을 목표로 삼지 마라. 기존 결과와 올바른 구현을 재사용하고, 불필요한 학습·렌더링·중복 계산부터 제거하라.

예상 장비는 **RTX 4090 24GB 1장, Intel i7-14700KF 20코어/28스레드, RAM 약 62GiB**다. 단, 반드시 실제 실행 환경에서 CPU affinity, 사용 가능한 RAM, CUDA 동작, GPU 여유 메모리, 다른 작업의 점유 상태를 확인하라. 어떤 샌드박스에서 CUDA가 안 보였다는 이유로 실제 워크스테이션도 CPU-only라고 판단하지 마라.

수학적 기준 원고는 첨부된 **`Ideal_Type_Finder(3).pdf`**, 40쪽, 제목 **Avoiding Resolution Collapse in Bayesian Preference Learning**이다. 특히 Section 3–5, Appendix A.10, Appendix B를 사용하라. 이 문서에는 필요한 여섯 알고리즘을 자체적으로 구현할 수 있도록 정의를 포함했다.

---

## 1. 가장 먼저 이해해야 할 실험의 의미

### 1.1 이번 실험은 RL 학습이 아니다

알고리즘은 연속적인 4차원 후보 좌표 q를 직접 합성한다. 가상 사용자는 고정된 이상점 theta_star를 가지며, 원고의 **음의 제곱거리 기반 multinomial logit**으로 답한다. 각 알고리즘은 자기 질문에 대한 자기 응답만 받아 posterior를 갱신한다.

Meta-World는 합성된 후보 q를 실제 simulator action sequence로 바꾸어 실행하는 역할을 맡는다. 따라서 이번 실험은 다음 두 부분으로 분리한다.

1. **원고와 같은 좌표 기반 선호추정 실험:** likelihood, posterior, query selector, integrated EIG ratio, posterior covariance trace.
2. **로봇 동작 구현 검사:** q를 실제 action으로 전달할 수 있는지, simulator에서 실행되는지, 서로 다른 q가 실제로 다른 motion으로 나타나는지.

SAC, PPO, PEBBLE, reward neural network, meta-RL, policy training, 매 질문마다의 trajectory optimization을 추가하지 마라. 기존 trajectory catalog에서 nearest-neighbor retrieval하는 문제로 바꾸지 마라.

### 1.2 결과 해석에서 반드시 지켜야 할 제한

이번 oracle은 실제 trajectory feature가 아니라 **원래 q 좌표**로 응답을 생성한다. 따라서 동일한 통계적 random stream이면 simulator 실행 여부와 무관하게 posterior와 EIG/MSE 곡선이 같아야 한다. q가 실제 로봇을 움직이는 파라미터여도 이 사실은 바뀌지 않는다.

이에 따라:

- 기존 synthetic history를 조건에 맞게 재사용할 수 있다. 재사용 사실은 명시한다.
- CPU에서 trajectory를 뒤늦게 병렬 replay해도 된다. 이 경우 **realization audit**라고 설명한다.
- 같은 통계 곡선을 “로봇 동역학에서 새로 얻은 독립적인 성능 향상”이라고 설명하면 안 된다.
- simulator에서 실행되었다고 collision-free, 실제 로봇 안전, task 성공, 사람의 구별 가능성, 모든 history에서의 theorem 가정까지 보장된다고 말하면 안 된다.
- q에서 action으로 가는 mapping이 full-rank여도 실제 위치 trajectory는 saturation이나 둔감한 축 때문에 q를 구분하지 못할 수 있다. 이를 별도 검사한다.
- 이 문제를 숨기기 위해 응답 utility를 trajectory feature로 바꾸지 마라. 그렇게 하면 요청한 통계 모델과 실험이 달라진다.

허용되는 설명:

> 원고의 4차원 ideal-point response model을 유지하고, 그 모델에서 합성된 질문들을 Meta-World motion clip으로 실제 실행하여 command 전달 정확도, 실행 유효성, motion 구분 정도를 검사했다.

금지되는 설명:

> Meta-World에서 Theorem 1/2가 증명되었다. 임의의 물리적 제약에서도 같은 rate가 보장된다. RL return이 좋아졌다.

### 1.3 무엇이 원고에서 왔고 무엇이 새 설계인지 구분

`PROVENANCE.md`에 세 부분을 작성하라.

- **원고에서 유지:** likelihood, 여섯 selector 정의, scale grid, Gaussian cap, EIG reference, covariance-trace metric, posterior numerical target.
- **사용자 지정 변경:** d=4 고정, T=128, 5 interactions, 여섯 알고리즘만 실행, beta sweep도 d=4에서 실행. 원고의 beta sweep은 d=8이므로 이 차이를 숨기지 마라.
- **이번 구현 설계:** 50-action clip, 네 개 motion basis, CPU/GPU 병렬 처리, trajectory replay, 물리적 진단, seed별 곡선 표시.

### 1.4 특히 중요한 SG 제한

원고의 SG 스케줄은 다음과 같다. t는 지금까지 받은 답의 수다.

\[
j(t)=\lfloor\log_2(1+t/128)\rfloor,
\qquad s_t=2^{-j(t)}.
\]

T=128이면 실제 질문은 t=0,...,127에서 생성된다. 따라서 **모든 실제 질문에서 j(t)=0, s_t=1**이다. stage 시작 때 저장한 중심도 이 구간에서는 바뀌지 않는다.

이번 짧은 실험에서 SG의 scale collapse가 안 보이는 것은 구현 오류가 아닐 수 있다. 원래 스케줄을 임의로 압축하지 마라. SG가 반드시 128회 안에 무너져야 한다는 검사나 그림 조작을 넣지 마라. SG를 posterior-coupled baseline이라고 부르지 마라. 별도 SG-fast 변형은 이번 요청 범위에 없다.

### 1.5 5 seeds의 해석

5 interactions는 고정된 계산 예산이지 유의성이나 충분한 검정력을 보장하는 숫자가 아니다. 모두 사용하고, 각 interaction 곡선과 평균을 제시한다. 결과가 불리한 seed를 버리거나, ranking이 애매한 setting에만 seed를 더 추가하지 마라. 128 rounds를 128개의 독립 표본으로 계산하지 마라. T=128 결과로 asymptotic rate를 입증했다고 쓰지 마라.

---

## 2. 변경하지 말아야 할 최종 설정

### 2.1 정확히 일곱 setting

모든 setting에서 d=4다.

| setting_id | M | beta | 용도 |
|---|---:|---:|---|
| m02_b04 | 2 | 4 | M sweep |
| m04_b04 | 4 | 4 | 두 sweep의 공통 기준점, 한 번만 실행 |
| m08_b04 | 8 | 4 | M sweep |
| m16_b04 | 16 | 4 | M sweep |
| m04_b02 | 4 | 2 | beta sweep |
| m04_b08 | 4 | 8 | beta sweep |
| m04_b16 | 4 | 16 | beta sweep |

모든 setting에서 **RC, PQ, QI, PS, GC, SG**를 실행한다. 각 알고리즘/setting에 interaction ID `[0,1,2,3,4]`를 사용한다. 기존 결과의 prefix를 재사용할 경우에는 대응 ID와 실제 seed를 비교 결과를 보기 전에 확정한다.

oracle beta와 inference beta는 항상 같다. 이것은 **정확하게 지정된 모델의 response sensitivity sweep**이지 beta misspecification 실험이 아니다.

### 2.2 공통값

- Prior: uniform on K=[-1,1]^4.
- theta_star: `(setting_id, interaction_id)`마다 한 번 추출한 뒤 128회 동안 고정.
- 같은 interaction의 여섯 알고리즘은 같은 theta_star를 공유한다.
- posterior, query randomness, answer stream, inference random stream은 알고리즘별로 분리한다.
- Scale grid: `[0.125,0.2,0.32,0.5,0.8,1.25,2.0]`.
- Gaussian physical cap: 현재 posterior mean 중심, 반지름 32.
- 후보 q는 prior support K 밖으로 나가도 된다. q를 [-1,1]^4로 clip하지 마라.
- 최종 답변 수: 각 run 정확히 128.
- 주 지표: realized-query integrated-EIG ratio와 모든 4좌표의 posterior covariance trace.

### 2.3 계산해야 할 작업 개수

재사용 및 정확한 cache 적용 전 기준:

- Statistical runs: 7 × 6 × 5 = **210**.
- Actual preference answers: 210 × 128 = **26,880**.
- 일곱 setting의 M 합: 2+4+8+16+4+4+4 = **42**.
- Selected-candidate logical rollout requests: 5 × 128 × 6 × 42 = **161,280**.
- 각 clip H=50이면 **8,064,000 environment action steps**.

초기화, 검증, reference audit, 영상 생성 비용은 별도다. `env.step` 한 번과 MuJoCo physics step 한 번을 혼동하지 마라. 설치된 환경의 `frame_skip`/substeps를 확인하여 각각 기록한다. 정확한 cache hit로 실제 실행 수가 줄어들면 logical requests와 unique executions를 구분한다.

---

## 3. 코딩 전에 기존 repository부터 확인

현재 repository의 `AGENTS.md`, git 상태, 실행 중인 관련 작업을 확인하라. 기존 변경사항과 실험을 덮어쓰거나 지우지 마라. 새 작업은 `metaworld_fast/` 아래에 격리하는 것을 권장한다.

기존 구현에서 다음을 찾아라.

- RC-MLQ/Fisher score/Posterior Quantiles.
- PQ/QI/PS/GC/SG/GaussCloud.
- adaptive marginal quadrature, resample-move SMC.
- 257 nodes, 4096 particles, 원고의 scale grid와 SG schedule.
- 저장된 실제 q/y history, posterior summaries, EIG reference, seed manifest.

이미 올바른 selector와 inference가 있으면 재사용하라. 이번 작은 실험을 위해 repository 전체를 먼저 리팩터링하지 마라.

### 3.1 가장 먼저 적용할 가속: 기존 synthetic prefix 재사용

다음이 모두 일치할 때만 기존 history의 첫 128개 답을 사용할 수 있다.

1. d, M, beta, prior, scale grid, cap, selector, numerical engine.
2. 첫 128회까지의 스케줄·tie rule·adaptation이 원래 terminal budget에 의존하지 않음.
3. 미래 데이터를 사용한 smoothing posterior가 아님.
4. 그림 이미지가 아니라 실제 질문/답/수치 데이터가 있음.
5. 사용할 첫 다섯 interaction을 성능과 무관하게 미리 정함.
6. 필요한 numerical validation을 재사용하거나 추가함.
7. 원본 경로, code/config hash, 실제 seed, 재사용 범위를 기록함.

원고의 **d=4, beta=4 M sweep**은 재사용 후보가 될 수 있다. **d=8 beta sweep을 d=4로 재표기하면 안 된다.** d=4에 맞는 beta 결과가 없으면 새로 실행한다.

공통 `(M=4,beta=4)`는 중복 실행하지 마라. 두 figure에서 사용하더라도 independent interaction이 열 개가 되는 것은 아니다.

---

## 4. 통계 모델: 잘못 구현하기 쉬운 부분까지 명확히

### 4.1 기호

- theta_star: 가상 사용자의 고정된 이상점.
- theta: learner가 가능한 값으로 생각하는 preference parameter.
- q_m: 화면에 대응시킬 후보의 4차원 좌표. reward weight나 robot joint vector가 아니다.
- Q=(q_1,...,q_M): 순서가 있는 한 질문의 후보 tuple.
- p_t: t개의 답 이후 posterior.
- mu_t, Sigma_t: posterior mean/covariance.
- sigma_t: 최대 고유값의 제곱근.
- v_t: posterior variance가 가장 큰 단위 방향.

### 4.2 원고와 동일한 likelihood

\[
\ell_m(\theta,Q)=
\frac{\exp[-\beta\|\theta-q_m\|^2]}
{\sum_j\exp[-\beta\|\theta-q_j\|^2]}.
\]

안정적인 구현에는 공통항을 지운 다음 logits를 사용해도 된다.

\[
z_m=2\beta\theta^\top q_m-\beta\|q_m\|^2.
\]

**후보마다 다른 -beta||q_m||² 항을 절대 빼지 마라.** `softmax(beta*theta@q)`는 다른 모델이다.

`log_softmax`/`logsumexp`를 사용하고 natural log로 계산한다. Particle predictive probability의 로그 역시 `logsumexp(log_weights + log_likelihood, particles)`로 계산하라. 이미 0으로 underflow한 확률 합에 log를 취하지 마라.

Oracle은 `ell(theta_star,Q_displayed)`에서 categorical answer 하나를 추출한다. theta_star는 oracle과 오차 평가기에만 전달하고 selector나 robot adapter에는 전달하지 마라.

### 4.3 Posterior update

실제 답이 y라면:

\[
p_{t+1}(\theta)\propto p_t(\theta)\ell_y(\theta,Q_{t+1}).
\]

이 likelihood를 target으로 하는 numerical Bayesian update를 수행한다. Gaussian assumed-density update, mean-only update, noiseless nearest-candidate answer로 대체하지 마라.

### 4.4 Moments와 MSE

\[
\mu_t=\mathbb E_{p_t}[\Theta],\qquad
\Sigma_t=\mathbb E_{p_t}[(\Theta-\mu_t)(\Theta-\mu_t)^\top],
\qquad \sigma_t=\sqrt{\lambda_{\max}(\Sigma_t)}.
\]

한 interaction의 conditional posterior risk는

\[
R_t=\operatorname{tr}(\Sigma_t)
\]

이다. 독립 interaction들에 대해 이를 평균하면 원고의 Bayes MSE를 수치적으로 추정한다. 별도로 actual squared error도 저장하라.

\[
e_t=\|\mu_t-\theta^\star\|^2.
\]

R_t와 e_t는 한 interaction에서 동일한 값이 아니다. 정확한 초기 prior는 mu_0=0, Sigma_0=I_4/3, R_0=4/3이므로 검증에 사용한다. Particle approximation에는 별도 numerical error가 있다.

### 4.5 Principal direction과 tie rule

정확히 factorized된 posterior에서는 가장 큰 marginal variance의 coordinate를 선택한다. 동률이면 가장 작은 index, 양의 방향을 택한다.

Joint posterior에서는 full 4×4 covariance를 사용한다. 최대 고유공간의 projector를 P라고 할 때, P e_i가 0이 아닌 가장 작은 i를 찾아 정규화한 방향을 택한다. 수치적인 eigenspace tolerance를 명시하라. Full covariance를 대각행렬로 바꾸거나 동률 방향을 임의 랜덤으로 정하지 마라.

Roundoff 제거용 대칭화는 가능하지만, 유의미한 음의 고유값·zero width를 임의 ridge나 spread floor로 숨기지 마라. Numerical failure를 진단하고 posterior 계산을 보강한다.

---

## 5. 여섯 알고리즘의 완전한 정의

### 5.1 RC/PQ 및 reference가 사용하는 line template

Posterior를 v_t 방향으로 projection하고 mean을 뺀 변수를 S_t라고 한다.

\[
S_t=v_t^\top(\Theta-\mu_t).
\]

c_t는 S_t의 median이다. xi_{t,m}는 확률 `(m-0.5)/M`의 quantile이며 m=1,...,M이다. 다음처럼 posterior width를 제거한다.

\[
b_{t,m}=(\xi_{t,m}-c_t)/\sigma_t.
\]

Physical coordinate scale r에 해당하는 후보는

\[
q_{t+1,m}(r)=\mu_t+(c_t+r b_{t,m})v_t
\]

이다.

Median과 quantile은 실제 posterior representation에서 계산하라. 임의로 Gaussian quantile로 대체하지 마라. Marginal quadrature는 continuous CDF를 invert한다. Particle posterior는 projected coordinates를 정렬하고 normalized weights의 누적합이 목표 확률을 처음 넘는 좌표를 weighted quantile로 사용한다. Repeated quantile/zero empirical width는 진단 대상이며 임의 후보 교체로 숨기지 마라.

### 5.2 RC: Resolution-Calibrated Multiway Line Queries

각 scale r에서:

\[
\omega_m(r)=\operatorname{softmax}_m(-\beta r^2 b_{t,m}^2).
\]

Fisher score는 다음과 같다.

\[
J_t(r)=4\beta^2r^2
\left(\sum_m\omega_m(r)b_{t,m}^2
-\left[\sum_m\omega_m(r)b_{t,m}\right]^2\right).
\]

이 score는 **mu_t+c_t v_t인 projected median center**에서의 score다. Median과 mean을 혼동하지 마라.

고정된 7개 scale에서 J_t를 최대화하고 동률이면 작은 r을 고른다. 그 r의 line query를 실제 질문으로 사용한다.

RC의 선택을 exact EIG maximization으로 바꾸지 마라. Scale grid를 beta/round/posterior width에 따라 바꾸지 마라. Fisher 계산에 rollout은 필요 없다. Template, quantile, 모든 scale score를 공유·cache한다.

### 5.3 PQ: Posterior Quantiles

동일한 line template에서 r=sigma_t를 사용한다.

\[
q_{t+1,m}^{PQ}=\mu_t+\xi_{t,m}v_t.
\]

독립 posterior sample을 뽑는 것이 아니다. Minimum scale, spacing floor, RC fallback, 추가 clipping을 넣지 마라.

### 5.4 QI: 한 번의 hypothetical-posterior-mean transformation

1. **QI 자신의 현재 posterior**에서 Fisher-best RC seed question을 만든다.
2. 같은 seed question의 가능한 답 m 각각에 대해 다음을 계산한다.

\[
\bar\ell_m=\mathbb E_{p_t}[\ell_m(\Theta,Q_{seed})],
\qquad
\mu^{(m)}=
\frac{\mathbb E_{p_t}[\Theta\ell_m(\Theta,Q_{seed})]}{\bar\ell_m}.
\]

3. 실제 질문을 `(mu^(1),...,mu^(M))`으로 만든다.
4. 이 실제 질문의 실제 답 하나만 posterior에 반영한다.

Inner transformation은 정확히 한 번이다. 모든 hypothetical answer는 같은 frozen posterior에서 출발한다. 가상 답을 차례대로 실제 posterior에 update하지 마라. 별도로 돌아가는 RC run의 posterior를 가져오지 마라. Convergence까지 반복하거나 중복 후보를 교체하지 마라.

Factorized posterior의 coordinate-aligned seed이면 선택 coordinate만 hypothetical mean이 바뀌고 나머지는 그대로다. 이것은 exact cancellation을 이용하는 계산 단축이다.

### 5.5 PS: Independent Posterior Candidates

현재 posterior approximation에서 M개의 **joint vector를 독립적으로, 복원추출**한다. Weighted particle이면 normalized particle weights로 index를 추출한다.

Repeated draw와 후보 slot을 유지한다. Joint posterior의 coordinate를 독립 marginal처럼 따로 샘플링하지 마라. Deduplication, without-replacement sampling, diversity filter, principal-axis restriction, catalog retrieval을 넣지 마라.

### 5.6 GC: Posterior-Covariance Gaussian Cloud

독립적인 z_m~N(0,I_4)에 대해:

\[
q_m^{raw}=\mu_t+\Sigma_t^{1/2}z_m.
\]

Covariance multiplier는 1이다. Full covariance를 사용한다. `sigma_t² I` 또는 covariance diagonal만 사용하면 다른 baseline이다. 아래의 반지름 32 cap을 적용한다. 후보를 한 직선으로 제한하지 마라.

### 5.7 SG: Scheduled Gaussian Cloud

\[
j(t)=\lfloor\log_2(1+t/128)\rfloor,
\qquad s_t=2^{-j(t)}.
\]

새 stage에 들어갈 때 현재 posterior mean을 stage center로 저장한다. Stage 내부에서는 center를 고정한다.

\[
q_m^{raw}=o_{stage}+s_t z_m,\qquad z_m\sim N(0,I_4).
\]

마지막에 cap을 적용한다. **Cap 중심은 saved stage center가 아니라 현재 posterior mean**이다.

Boundary tests: t=0,127은 stage 0; t=128,383은 stage 1; t=384는 stage 2. 최종 T=128 실험의 실제 질문은 전부 stage 0이다. Final posterior risk를 저장하기 위해 t=128의 dummy query를 만들지 마라. 그것은 관측하지 않은 새 stage를 시작시킨다.

### 5.8 GC/SG의 fixed physical cap

각 candidate의 delta=q_raw-mu_t에 대해:

\[
q_{display}=\mu_t+
\min\left(1,\frac{32}{\|\delta\|}\right)\delta.
\]

delta=0이면 그대로 유지한다. Raw와 displayed q를 둘 다 저장한다. 답 생성, actual-query EIG, posterior update는 **displayed q**를 사용한다.

이는 radial projection이다. Coordinatewise clipping이나 posterior-width 비례 cap이 아니다. RC에는 이 projection을 적용하지 마라. RC는 원래 query를 변경하지 않고 범위를 검사한다. PS와 QI에도 불필요한 변형을 추가하지 마라.

---

## 6. 지표 계산: Fisher를 EIG라고 부르면 안 된다

### 6.1 Integrated Expected Information Gain

\[
\bar\ell_{t,m}(Q)=\mathbb E_{p_t}[\ell_m(\Theta,Q)],
\]

\[
I_t(Q)=\mathbb E_{p_t}\left[
\sum_m\ell_m(\Theta,Q)
\log\frac{\ell_m(\Theta,Q)}{\bar\ell_{t,m}(Q)}
\right].
\]

Weighted nodes/particles `(theta_i,w_i)`에서는 적분을 weighted sum으로 계산한다.

\[
\bar\ell_m\approx\sum_iw_i\ell_{im},
\qquad
I_t(Q)\approx\sum_iw_i\sum_m\ell_{im}
(\log\ell_{im}-\log\bar\ell_m).
\]

J_t, `0.5*sigma_t²*J_t`, predictive entropy, 선택된 답 하나의 entropy, Fisher determinant는 이 EIG와 다르다. 보고 지표로 대체하지 마라.

### 6.2 작은 EIG의 수치 안정성

Likelihood/moments/EIG의 reference 계산은 float64로 한다. 비슷한 두 entropy를 빼는 구현만 사용하지 마라.

각 particle에서 `a_m=log(ell_im)-log(pbar_m)`라 두면 다음 KL identity를 사용할 수 있다.

\[
\operatorname{KL}(\ell_i\|\bar\ell)
=\sum_m\bar\ell_m f(a_m),
\qquad f(a)=e^a a-\operatorname{expm1}(a).
\]

작은 |a|에서는:

\[
f(a)=a^2/2+a^3/3+a^4/8+a^5/30+a^6/144+\cdots.
\]

이 식은 확률 차의 선형항 합이 0이라는 사실에서 나온 안정화 방법이다. 큰 |a|에서는 `pbar*exp(a)`가 overflow/underflow하지 않도록 이미 계산된 ell, pbar와 finite log ratio를 사용한 동치식을 택하라. Predictive log probability부터 log-domain으로 계산해야 한다. 직접 KL 및 작은 고정밀 예제와 비교하여 검증한다.

유의미하게 음수인 EIG를 0으로 clamp해 숨기지 마라. Numerical error인지 판단하고 보강한다.

### 6.3 각 방법 자신의 posterior에서 만든 reference

각 알고리즘의 매 round마다 **그 알고리즘 자신의 posterior**로 principal direction/quantile template과 7개 reference question을 만든다.

\[
I_t^\star=\max_{r\in\mathcal S}I_t(Q_{t+1}(r)),
\qquad
\eta_t=I_t(Q_{t+1}^{display})/I_t^\star.
\]

PS/GC/SG의 실제 질문은 full-dimensional이어도 reference는 자신의 joint posterior에서 생성한 line family다.

Off-grid method는 eta>1이 될 수 있다. 분모는 모든 질문의 전역 최댓값이 아니라 7개 grid의 최댓값이다. 값을 1로 clip하지 마라. RC는 동일 grid에서 하나를 선택하므로 numerical tolerance를 제외하면 eta<=1이어야 한다.

분모가 numerical error 대비 너무 작아 unresolved이면 frozen history에서 적분을 보강하고 flag를 남긴다. Arbitrary denominator floor를 넣어 유효한 ratio인 것처럼 만들지 마라. Unresolved seed를 몰래 평균에서 제외하지 마라.

### 6.4 시간 indexing

t=0,...,127마다:

1. p_t의 moments 계산.
2. Q_{t+1} 선택 및 **p_t에서** eta_t 계산.
3. q와 rollout request 저장.
4. 실제 oracle answer 추출.
5. p_{t+1}로 update.

Risk/actual squared error는 t=0,...,128의 129개 상태를 저장한다. Eta는 t=0,...,127의 128개 값을 저장한다. 자기 질문의 답을 반영한 뒤 그 질문의 EIG를 평가하지 마라.

### 6.5 Randomized-query diagnostics

PS/GC/SG에 대해 pre-answer t=`[0,31,63,95,127]`에서 posterior를 freeze하고 64개의 독립 diagnostic question을 추가 추출한다. 실제 답은 받지 않는다. 평균 integrated EIG를 같은 reference로 나누어 conditional query-averaged ratio를 추정한다.

Monte Carlo SE는 별도 저장한다. SG schedule state와 RNG를 clone하라. 이 64개는 추가 interaction이나 사용자 응답이 아니며 posterior를 바꾸면 안 된다. 기본적으로 simulator replay는 하지 않고 coordinate-model diagnostic이라고 표시한다.

Main figure는 이 conditional average가 아니라 실제 질문의 eta를 다섯 interaction에 대해 평균한 값이다.

---

## 7. Posterior engine: 정확한 계산 구조를 유지하면서 가속

### 7.1 RC/PQ/QI: exact coordinate factorization 사용

Product uniform prior에서 RC/PQ/QI가 coordinate-aligned question을 생성하면 likelihood에서 나머지 좌표 기여가 취소되어 posterior가 네 marginal의 product로 유지된다.

원고 구현에 맞춰:

- Marginal마다 처음 257 deterministic nodes.
- Log-domain updates.
- `x=c+s*sinh(u)` 형태의 adaptive mesh로 [-1,1] 지원; u에서 적분하면 Jacobian을 포함.
- Simpson integration 또는 검증된 동등한 adaptive numerical integration.
- Integrated continuous CDF inversion으로 quantile 계산.
- Remesh 시 해당 coordinate의 **전체 실제 likelihood history**를 다시 평가.
- 257 vs 513 nodes 등 doubled-resolution frozen-history checks.

Factorization 여부는 method 이름이 아니라 **실제로 표시한 q**를 보고 검사한다. 한 질문에서 둘 이상의 coordinate가 후보별로 달라지면 product update를 적용하지 마라. 오류를 진단하거나 명시적으로 joint engine을 사용한다.

257^4 tensor grid를 만들지 마라. PS/GC/SG에 이 shortcut을 억지로 적용하지 마라. 고정된 coarse grid로 좁아진 posterior를 놓친 채 실험을 끝내지 마라. Adaptive mesh가 기존에 없다면 정확도 검사를 통과하는 1D positive-density adaptive integration으로 구현하고 원고 구현과 차이를 기록한다.

### 7.2 PS/GC/SG: joint tempered resample-move SMC

최종 실험은 **4096 particles**를 사용한다. 더 작은 N은 plumbing smoke test에서만 사용하고 final 데이터에 섞지 마라.

원고에 맞는 요구사항:

- 새 likelihood를 fractional exponent로 순차 반영하되 최종 exponent는 반드시 1.
- Tempering increment는 ESS>=0.8N을 목표로 정함.
- Tempering stall에는 intermediate resampling으로 대응.
- Resampling마다 **10 Metropolis sweeps**.
- Target은 prior × 이전 전체 likelihood × 현재 새 likelihood의 fractional power.
- K=[-1,1]^4 밖 proposal은 reject. Boundary로 clip하지 않음.
- 최종 weights는 ESS<0.5N이 아니면 유지.
- ESS, increment, resampling, proposal acceptance, timing 기록.

Preconditioned random-walk covariance는 sweep 내에서 freeze한다. Adaptation은 frozen sweeps 사이에서 수행하고 기록한다. Asymmetric proposal을 쓰면서 Hastings correction을 빠뜨리지 마라.

Gaussian approximation을 **delayed-acceptance screen**으로 사용할 수 있지만 posterior target을 Gaussian으로 바꾸면 안 된다. Symmetric proposal과 고정된 screen density g의 경우:

\[
\alpha_1=\min(1,g(y)/g(x)),
\qquad
\alpha_2=\min\left(1,\frac{\pi(y)g(x)}{\pi(x)g(y)}\right).
\]

여기서 pi는 실제 tempered posterior다. Log-space로 구현하고 stage 1을 통과한 proposal에만 exact full-history correction을 계산한다. g는 해당 sweep에서 freeze한다. 기존 검증된 delayed-acceptance kernel을 우선 재사용하라.

금지: initial particles를 계속 reweight만 하기, rejuvenation 생략, 10 sweeps를 몰래 줄이기, 새 likelihood를 일부만 반영하기, Gaussian covariance를 정확 posterior covariance라고 부르기.

### 7.3 Numerical checks

가능하면 기존 원고 검증 artifacts를 재사용하라. 없으면:

- d=2, T=32, M={2,3,4}의 작은 interaction을 dense/adaptive quadrature와 대조.
- Final setting 중 대표 frozen histories를 답 수 {0,32,64,128}에서 검사. beta=16, M=16을 포함.
- 같은 q/y history를 8192 particles로 독립 replay하고 unresolved이면 reference를 추가 보강.
- Mean, full covariance, risk, eta 비교.
- Marginal variance/EIG relative tolerance 0.005와 ratio absolute tolerance 0.005를 유지. SMC에서는 stochastic reference uncertainty도 별도로 추정하여 판정하고, 한 번의 noisy reference를 exact truth로 취급하지 마라.
- Fixed-history check에서는 새 adaptive query를 다시 만들지 마라. 동일한 관측 history에서 적분 정확도를 비교해야 한다.
- 검증 history는 ranking을 보기 전에 고정한다. Numerical failure에 따른 추가 검사와 성능이 불리해서 seed를 바꾸는 행위를 구분한다.

SMC reference를 정교하게 해도 원래 N=4096 approximation의 문제가 해결되지 않을 수 있다. 이때 unresolved 상태를 숨기지 말고 원인과 추가 비용을 보고하라. 정확도를 무시하거나 무한 재시도하지 마라.

---

## 8. Meta-World query adapter

### 8.1 Task와 d=4의 의미

앞서 논의한 **reach-v3 한 task**를 기본으로 사용한다. 설치된 공식 Meta-World API와 scripted expert를 활용한다. 새 RL policy를 학습하지 마라.

Action interface는 `(dx,dy,dz,gripper)`의 4차원이지만, reach에서 q_4를 의미 없는 gripper 신호로만 쓰면 4차원 motion이 실제로 표현되지 않을 수 있다.

따라서 이번 adapter는 **3개 Cartesian channel의 시간축 위에 4개 독립 smooth basis를 두고 gripper는 고정**한다. 이것은 원고가 정의한 내용이 아니라 이번에 명시한 구현 선택이다. 통계적 d=4는 유지되지만 “q의 네 좌표가 순간 action의 네 좌표와 1:1 같다”고 설명하지 마라.

### 8.2 H=50과 공통 initial context

각 candidate는 **50 environment action steps**의 짧은 motion clip이다. Full task 성공은 요구하지 않는다. 가능한 공식 옵션으로 success 시 조기 종료를 끄되, 실제 terminal/truncation 이후에 계속 step하지 마라. 예상보다 짧은 clip은 flag를 남긴다.

다섯 interaction에 대응하는 physical context 다섯 개를 별도 seed로 생성한다. 같은 context ID는 모든 method, M, beta에서 공유한다. 같은 query의 모든 후보는 반드시 같은 initial physical state와 goal에서 시작한다.

각 context에서:

1. Task parameters, goal, simulator와 wrapper의 필요한 전체 상태를 저장.
2. Official scripted expert로 nominal action sequence 50개를 한 번 생성.
3. 실제로 environment에 전달한 bounded actions를 저장. Expert raw output이 처음부터 bounds 안이라고 가정하지 마라.
4. 모든 candidate를 동일 context로 restore한 후 실행.

이 nominal sequence를 cache한다. 매 candidate의 매 step마다 expert를 다시 호출하지 마라. 이번 정의는 저장된 nominal sequence에 대한 open-loop perturbation이다. Perturbed state에서 feedback expert를 다시 계산하는 방식으로 몰래 바꾸지 마라.

### 8.3 네 개 motion basis

h=0,...,H-1, u_h=h/(H-1)로 두고:

\[
f_h=\sin(\pi u_h),\qquad g_h=\sin(2\pi u_h).
\]

Cartesian 단위벡터 e_x,e_y,e_z를 이용하여:

\[
B_1(h)=f_h e_x,\quad B_2(h)=f_h e_y,\quad
B_3(h)=f_h e_z,\quad B_4(h)=g_h e_x.
\]

이를 flatten한 (3H)×4 행렬 B의 rank와 condition number를 검사한다. q_4는 gripper가 아니라 다른 시간 패턴의 x-motion을 조절한다.

Baseline Cartesian command는 bounded nominal command의 0.6배로 두고, gripper는 검증한 고정값으로 유지한다. Candidate q의 command는:

\[
a_h^{xyz}(q)=a_{base}(h)+\alpha\sum_{k=1}^4q_kB_k(h).
\]

이 action을 실제 simulator에 넣어 trajectory를 얻는다. 기존 state trajectories를 단순 가산·보간하여 “feasible rollout”이라고 만들지 마라.

### 8.4 모든 method에 같은 global gain

이번 여섯 method와 reference에서 `abs(q_i)<=33`을 보수적 envelope로 사용할 수 있다.

- GC/SG의 displayed q는 [-1,1]^4의 mean에서 반지름 32 이내.
- 원고 bounds로 sigma<=2, |c|<=2, |b_m|<=1+sqrt(2M-1).
- M<=16, r<=2이므로 RC reference의 coordinate bound는 `1+2+2*(1+sqrt(31))<33`.
- 나머지 지정 selector도 이 envelope 안에 있음.

이것은 이번 설정의 bound이며 모든 가능한 query policy에 대한 주장이 아니다. Runtime에서 확인하고 위반하면 오류를 조사한다. q를 clip하지 마라.

`L_B=max_(h,axis) sum_k abs(B_k(h,axis))`로 두고:

\[
\alpha=\frac{0.35}{33L_B}.
\]

그러면 Cartesian command의 magnitude는 최대 0.6+0.35=0.95다. Alpha는 모든 setting/seed/method/round에서 동일하게 고정하고 저장한다. Per-question normalization/rescaling은 금지한다.

이 conservative gain은 흔한 q에서 motion이 작아지는 단점이 있다. 이를 숨기거나 질문마다 gain을 키우지 마라. 실제 motion sensitivity를 검사하라. 다른 adapter가 필요하면 성능 ranking을 보기 전 engineering calibration으로 명시적인 새 version을 만들고 보고해야 한다. 통계적 q 자체를 로봇에 맞추어 바꾸지 마라.

Action Box 안에 있어도 internal workspace clipping/saturation은 가능하므로 별도 기록한다.

### 8.5 실제로 전달된 command에서 q 복원

Precompute한 B의 pseudoinverse로 실제 submitted action에서 q를 복원한다.

\[
\widehat q_{cmd}=B^+\operatorname{vec}((a^{xyz}-a_{base})/\alpha).
\]

Float32 action transport까지 포함하여 `||q_hat_cmd-q||`를 기록한다. 초기 tolerance는 1e-4로 두고 실제 round-trip error에 비추어 검증한다. 이 tolerance를 사람의 구별 가능성 기준으로 해석하지 마라.

Command 복원 성공은 executed trajectory의 invertibility를 보장하지 않는다.

### 8.6 실제 motion 진단

모든 selected candidate에 대해 다음을 저장하라.

- Finite state/action, action-bound violations, 실제 clipping 여부.
- Simulator exception/warning, 예상 밖 종료, 실제 실행 step 수.
- 전체 clip의 end-effector Cartesian positions. Simulator array의 view를 그대로 저장하지 말고 copy하라.
- 가능한 gripper state, final goal distance, task-success flag. 이것은 보조 진단이며 oracle utility가 아니다.
- Workspace/mocap saturation.
- Command reconstruction error.
- Context/query/adapter hash와 simulator version.

각 query에서 실제 위치 time series의 **pairwise RMS motion separation**을 계산하고 coordinate pair distance와 함께 저장한다. 이를 원고의 smallest-enclosing-ball radius a(Q)라고 부르지 마라. PS의 합법적인 duplicate 때문에 생긴 zero pair distance는 유지한다.

Full run 전에 고정된 calibration points에서 q 각 좌표의 finite difference를 여러 perturbation 크기로 검사하라. 전체 위치 time series의 (3H)×4 Jacobian singular values와 numerical rank를 보고한다. Final endpoint나 task success만으로 4차원 motion sensitivity를 판단하지 마라.

서로 다른 q가 사실상 같은 motion이 되거나 특정 coordinate가 보이지 않으면 physical interpretation을 **unresolved/failed**로 표시한다. 유효한 coordinate-model 계산은 보존하되 성공적인 robotics validation이라고 주장하지 마라. 문제 후보·seed를 몰래 제거하지 마라.

### 8.7 안전성과 실행 가능성 구분

Simulator가 finite state로 진행되었다고 collision-free나 실제 로봇 안전이 보장되지는 않는다. `simulation_valid`, `action_bounds_ok`, `workspace_saturation_rate`, `task_success`를 구분하라. 오해를 부르는 하나의 `safe=True` flag로 합치지 마라.

### 8.8 Reference grid의 물리적 audit

EIG reference를 계산할 때마다 7개 scale을 전부 rollout할 필요는 없다. 이는 coordinate-model 적분이다.

다만 모든 grid command의 action-bound 논증과 별도로, **interaction 0, t={0,127}**에서 일곱 setting·여섯 method의 reference grid를 실제 replay하여 같은 물리적 검사를 수행한다. 이것은 finite audit이고 universal feasibility proof가 아니다.

Randomized-query 64 draws는 기본적으로 coordinate diagnostic에만 사용한다. Main selected queries에는 전부 rollout request를 남긴다.

---

## 9. 가장 빠른 실행 구조

### 9.1 Statistical producer와 CPU rollout consumer를 분리

```text
기존 verified prefix 또는 새로운 inference
                    |
                    +--> q/y, posterior summaries, EIG/MSE
                    |
                    +--> deterministic rollout-request shards
                                       |
                             persistent CPU workers
                                       |
                         actual trajectories / exact cache
                                       |
                              검사 및 report merge
```

Oracle이 trajectory에 의존하지 않으므로 statistical loop가 매 candidate rollout이나 video를 기다릴 필요가 없다. CPU/GPU 자원 경합을 측정하며 inference와 rollout을 동시 진행하라.

Statistical completion과 physical-audit completion을 별도로 추적한다. 나중에 발견된 물리적 실패를 숨기거나 statistical history에서 제거하지 마라.

### 9.2 CPU

- 실제 physical/logical cores, affinity/cgroup, available RAM, 다른 작업 점유를 확인.
- Persistent workers 사용. Candidate마다 환경 생성·model compile 금지.
- Worker 수 후보 4,8,12,16,20,24 중 실제 사용 가능 범위에서 짧게 benchmark.
- Rollout worker 내부 BLAS/OpenMP threads는 1로 시작.
- 모든 worker가 각각 full-size BLAS pool을 만들지 않도록 방지.
- IPC는 complete query 또는 32–128 candidate chunk 단위. Step마다 통신하지 않음.
- M=16 작업이 끝에 몰리지 않게 queue 관리.
- 작은 marginal-inference CPU jobs도 persistent pool을 사용하거나 기존 fast code 재사용.
- 28 logical workers가 16/20 workers보다 빠르다고 가정하지 않음.

### 9.3 GPU

- 실제 CUDA tensor 연산과 synchronize로 동작 확인.
- 기본은 **CUDA를 소유한 inference process 하나**.
- Simulator workers는 CPU-only로 두고 CUDA context를 만들지 않음.
- 독립 SMC runs와 EIG evaluations를 batch화. M과 history 길이가 비슷한 run끼리 묶음.
- GPU batch size 8/16/32 등을 memory와 측정 throughput 기준으로 선택.
- Particles, weights, histories, proposal target log densities를 가능한 한 device에 유지.
- 4096 particles를 Python loop로 처리하지 말고 particles/candidates/scales/runs에 대해 vectorize.
- Full-history evaluation은 history/candidate 축을 chunk하여 temporary memory 폭증 방지.
- 질문별 `2*beta*q`, `-beta*||q||²`를 cache.
- 현재 particle의 target log likelihood는 재활용하고 proposal에 대해서만 새로 평가.
- Delayed-acceptance stage 1 통과분에만 exact history correction.
- CPU에는 작은 q batches와 summary만 전송. 매 round 전체 posterior를 복사하지 않음.
- 같은 posterior의 7개 scale은 projection/sorting/quantile을 공유.
- Pure inference에서 autograd/computation graph 비활성화.

Float64 작은 연산은 GPU가 반드시 빠르지 않다. CPU/GPU 두 경로를 실제 workload로 비교해 **검증을 통과한 더 빠른 경로**를 택하라. 이미 빠른 1D marginal routine을 GPU로 옮기는 데 실험보다 긴 시간을 쓰지 마라.

### 9.4 Precision

기준은 float64다. Likelihood/covariance/작은 EIG에 FP16, BF16, AMP, TF32를 무검증으로 적용하지 마라.

Float32 fast path는 어려운 setting의 frozen histories에서 numerical tolerance를 확인한 뒤에만 허용한다. Reductions 및 최종 metric은 float64로 유지할 수 있다. Adaptive histories가 달라지는 것을 bitwise 비교하지 말고 동일 history에서 계산 오차를 비교하라.

속도를 위해 particles, Metropolis sweeps, method definition을 몰래 줄이지 마라.

### 9.5 Process/thread 설정

Worker의 numerical library import 전에 다음을 설정할 수 있다.

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

CUDA를 초기화한 뒤 fork하지 마라. 적절한 spawn/forkserver와 `if __name__ == '__main__':`를 사용한다. GPU process의 CPU threads는 rollout worker와 별도로 튜닝한다. Worker scheduling이 RNG 결과를 바꾸면 안 된다.

### 9.6 전체 실험에서는 영상 렌더링 금지

Quantitative run은 `render_mode=None`, no GUI/viewer, no video encoder, no realtime sleep으로 실행한다. Numeric trajectories만 저장한다.

이후 미리 정한 작은 subset만 영상으로 만든다. 기본: M=4,beta=4, interaction 0, RC/PQ/SG, t={0,63,127}, 각 질문의 네 후보. 총 36 clips다. 예쁜 결과나 유리한 비교를 보고 골라서는 안 된다.

### 9.7 State restore와 low-level 가속

기준 구현은 동일 task/context를 복원하는 표준 env.step 경로다. Restore를 가속할 때는 replay equivalence를 먼저 검사한다.

qpos/qvel만 복원해서 충분하다고 가정하지 마라. 설치 버전에 맞춰 mocap, actuation, time, task/goal, mutable model fields, RNG, observation history, wrapper counters를 확인한다.

같은 context/action sequence의 replay가 candidate 순서와 worker 수가 바뀌어도 일치하는지 검사한다.

`mujoco.rollout`의 low-level actuator controls는 Meta-World의 4D Cartesian/mocap/gripper action과 자동으로 같지 않다. 그 함수에 4D actions를 바로 넣어 “동일한 빠른 환경”이라고 만들지 마라. Low-level executor를 쓰려면 transformations, substeps, workspace limits, observation updates까지 env.step과 비교 검증한다. 속도를 위해 다른 physics engine이나 미검증 MJX/Warp port로 바꾸지 마라.

---

## 10. 실제 runtime을 측정한 뒤 ETA를 산출

이전 대화의 몇 시간 추정치는 benchmark가 아니다. 반드시 이 워크스테이션에서 측정하라.

`PROFILE.json`, `ETA.md`에 다음을 분리한다.

- Environment initialization/context restore.
- Headless clip rollout, summary 및 저장 비용.
- Marginal inference, EIG reference, QI hypothetical integration.
- SMC update, full-history correction, sorting, diagnostics.
- Numerical validation replay.
- 그림·영상 생성.

Early history만 측정하지 말고 t≈64/128의 frozen histories와 M=16, beta=16을 포함한다.

### 10.1 튜닝 시간도 제한

올바른 smoke test가 된 뒤 worker/batch autotuning에 약 5–10분을 쓰고 불필요하게 길게 끌지 마라. 실제와 같은 workload로 비교한다. 무관한 대형 matrix multiplication 속도를 실험 속도라고 보고하지 마라.

CPU throughput이 v clips/sec이면 remaining uncached requests/v에 추가 비용을 더해 rollout ETA를 산출한다. Inference와 rollout이 동시 실행되면 전체 시간은 단순 합보다 두 remaining time의 max에 가까울 수 있으나, 자원 경합을 반영한 실제 결합 throughput을 사용해야 한다.

측정한 범위와 가정을 제시하고 진행 중 업데이트하라. 무근거한 “4090이니 몇 시간 안에 끝남”을 쓰지 마라.

### 10.2 가속 우선순위

1. 올바른 기존 prefix와 구현 재사용.
2. 공통 sweep anchor와 정확하게 같은 rollout request의 중복 제거.
3. Rendering, RL training, candidate마다 env 생성 제거.
4. RC/PQ/QI만 coordinate factorization 사용.
5. GPU joint-inference batching + 적절한 수의 persistent CPU workers.
6. Full-history coefficient/target cache + exact delayed acceptance.
7. 이후는 절감 예상 시간이 구현·검증 시간보다 큰 bottleneck만 최적화.

원래의 전체 d×M grid, 12개 binary methods, 4096-round suite를 함께 실행하지 마라.

---

## 11. RNG·재시작·데이터 무결성

### 11.1 RNG

Root integer, setting, interaction, method, purpose로 stable seed를 만든다. 고정된 정수 mapping 또는 cryptographic digest를 사용하고 Python의 randomized `hash()`를 쓰지 마라.

Theta stream에는 method를 넣지 않아 같은 interaction에서 theta_star를 공유한다. Response/selector/SMC/diagnostics는 purpose와 method별로 분리한다. Environment는 statistical outcomes에 의존하지 않는다.

기존 artifact를 재사용하면 실제 원래 seed/latent mapping을 보존하고 새 seed로 실행한 것처럼 표기하지 마라. Diagnostic draws와 videos가 actual query/answer RNG를 소비하면 안 된다.

### 11.2 Checkpoints

최소 16 answers마다 atomic checkpoint를 남긴다.

- 전체 actual q/y history.
- Nodes 또는 particles/weights.
- RNG states.
- SG stage와 center.
- Numerical adaptation 상태.
- Metrics, pending rollout request IDs.
- Config/code/adapter hashes.

Resume가 uninterrupted run과 일치하는지 검사한다. Statistical config가 바뀌면 다른 run ID를 만든다. 여러 process가 같은 파일에 안전장치 없이 append하지 마라.

### 11.3 정확한 rollout cache

Key에 exact displayed-q bytes와 dtype, complete physical context hash, H, basis, alpha, nominal actions, simulator/task version, adapter code hash를 포함한다.

q를 반올림하여 근처 trajectory로 대신하는 cache는 금지한다. 동일 key는 하나의 실행을 여러 logical request에 연결할 수 있지만, query의 duplicate candidate slots는 유지한다.

### 11.4 저장 형식

Chunked arrays/shards를 사용한다. Step마다 JSON/file 하나씩 만들지 마라. 모든 selected request의 positions와 compact diagnostics를 저장한다. Full observations 생략 시 actions/context/pinned simulator로 재현할 수 있어야 하고 저장 범위를 명시한다. Loop 안에서 무거운 compression을 하지 마라.

Planned command를 실제 executed rollout처럼 저장하지 마라. `pending`, `executed`, `cache_hit`, `failed`, `not_requested` 상태를 구분한다.

---

## 12. Full queue 전에 통과시킬 tests

### 12.1 통계 tests

- Probabilities finite/normalized.
- Squared-distance logits와 expanded logits가 일치; candidate norm offset 누락 감지.
- Theta와 모든 q를 같은 벡터만큼 이동하면 likelihood 불변.
- 동일 후보는 uniform probabilities 및 EIG=0.
- 후보 순서를 바꾸면 probabilities는 같은 순서로 바뀌고 EIG는 불변.
- 1D uniform prior, M=2에서는 mu=0, sigma=1/sqrt(3), c=0, xi=(-0.5,0.5), b=(-sqrt(3)/2,sqrt(3)/2). PQ=(-0.5,0.5). RC score=3*beta²*r²이므로 해당 대칭 예제에서 최대 grid scale 선택.
- QI hypothetical updates 동안 실제 posterior 불변; inner step은 한 번.
- PS는 weighted joint posterior에서 with replacement.
- GC는 full Sigma covariance.
- SG boundaries/center/state cloning 정확.
- Cap은 fixed radius, current mean 중심.
- 각 method의 reference는 자기 posterior.
- Factorization 실제 q로 검증.
- EIG∈[0,log M], RC eta<=1은 tolerance 내에서 성립; off-grid eta>1은 허용.
- 초기 full risk=4/3; 선택 coordinate variance만 risk로 보고하지 않음.
- CPU/GPU frozen-history 계산 일치.
- Diagnostics/worker scheduling이 actual histories를 바꾸지 않음.
- Resume와 uninterrupted run 일치.

### 12.2 Robot tests

- 설치된 action-space shape/bounds 확인.
- B rank=4, command reconstruction 검사.
- Context restore와 candidate-order invariance.
- 고정한 corner/axis/extreme-q command bounds와 simulator validity.
- 실제 위치 time-series의 네 coordinate sensitivity.
- Internal saturation과 너무 작은 motion 변화를 보고.
- Headless loop에 renderer가 생성되지 않음.
- Low-level fast path를 사용할 때 env.step equivalence 검증.

### 12.3 Manifest freeze

Engineering calibration으로 worker count/chunk size를 정하고 bugs를 수정할 수 있다. 하지만 method ranking을 보고 seeds, SG schedule, beta, gain, horizon을 선택하지 마라. Final manifest를 freeze하고 semantic change가 생기면 새로운 version으로 구분한다.

---

## 13. 구현 파일과 실행 interface

기존 repository에 맞춰 조정하되 다음 기능을 제공하라.

```text
metaworld_fast/
  README.md
  PROVENANCE.md
  LIMITATIONS.md
  config/final.yaml
  config/calibration.yaml
  model.py
  line_template.py
  selectors.py
  posterior_marginal.py
  posterior_smc.py
  metrics.py
  renderer.py
  contexts.py
  rollout_pool.py
  scheduler.py
  artifact_reuse.py
  validation.py
  reporting.py
  cli.py
  tests/
  scripts/run_detached.sh
  outputs/<run_id>/
    config_frozen.yaml
    manifest.json
    HARDWARE.json
    ENVIRONMENT.txt
    PROFILE.json
    ETA.md
    STATUS.md
    logs/
    checkpoints/
    statistical/
    rollout_requests/
    rollouts/
    diagnostics/
    validation/
    figures/
    videos/
    tables/
    REPORT.md
```

한 번의 명령으로 전체 실행/재시작을 지원하고 개별 단계도 실행할 수 있게 하라. 예시 interface:

```bash
python -m metaworld_fast.cli inspect --config metaworld_fast/config/final.yaml
python -m metaworld_fast.cli test --config metaworld_fast/config/final.yaml
python -m metaworld_fast.cli profile --config metaworld_fast/config/final.yaml
python -m metaworld_fast.cli run --config metaworld_fast/config/final.yaml --resume
python -m metaworld_fast.cli validate --run-dir <actual_run_dir>
python -m metaworld_fast.cli plot --run-dir <actual_run_dir>
python -m metaworld_fast.cli status --run-dir <actual_run_dir>
```

위 명령은 **구현해야 할 interface 예시**다. 파일이 이미 존재한다고 가정하지 마라. 실제로 실행해 검증한 명령만 사용자에게 사용 가능하다고 보고하라.

Linux SSH 환경에서는 검증된 tmux/nohup launcher, unbuffered logs, PID, signal handling, checkpoint를 제공하여 연결 종료 후에도 유지되게 한다. 자신이 시작한 process만 관리하라. SSH host나 외부 서버를 임의로 정하지 마라.

---

## 14. 최종 그림·표·해석

### 14.1 원고와 대응하는 네 statistical figures

1. M={2,4,8,16}, beta=4: eta curve.
2. 같은 M sweep: full posterior-risk/Bayes-MSE estimate.
3. Beta={2,4,8,16}, M=4: eta curve.
4. 같은 beta sweep: full posterior-risk/Bayes-MSE estimate.

모두 d=4다. Method 색/선/순서를 일관되게 유지한다. 다섯 individual curves를 얇게, arithmetic mean을 뚜렷하게 표시한다. 너무 복잡하면 method별 supplementary panels도 저장하라.

5 interactions에 자동으로 bootstrap 95% band나 p-value를 붙이지 마라. 표에서 SE를 쓰면 descriptive between-interaction standard error임을 명시한다.

Raw curves를 사용한다. Separation을 만들려고 smoothing하거나 eta>1을 지우지 마라. 128회를 넘어 extrapolate하지 마라. T^-1, T^-1/2 guide를 넣으면 fitted/established rate가 아닌 **reference slope**로 명시한다. SG의 첫 stage가 고정되어 있다는 설명을 caption에 넣는다.

Vector PDF와 PNG를 저장하고 실제 출력의 글자·legend·잘림을 확인한다. 기존 plotting conventions를 재사용하되 불필요한 dependency를 추가하지 마라.

### 14.2 Meta-World가 실제로 추가하는 결과

다음은 생략하면 안 된다.

- Method/setting별 physical-audit coverage와 validity table.
- Command encoding/reconstruction error.
- Executed-motion separation과 q-coordinate separation 비교.
- Saturation/failure rates와 motion-Jacobian singular-value diagnostics.
- 미리 정한 subset의 실제 robot clips.
- Reference-grid audit 범위와 unresolved cases.

이 진단은 주 지표 두 개를 대체하는 것이 아니라 추가하는 것이다.

### 14.3 Tables

답 수 32/64/128의 risk와 realized squared error, 대응 pre-answer t=31/63/127의 eta를 seed별·평균으로 저장한다. Numerical discrepancies, 실제 runtime, logical/unique rollout counts를 함께 export한다.

매 평균의 유효 interaction 수를 기록한다. 일부 run이 실패했는데 네 개만 평균하여 다섯 개 완료한 것처럼 표시하지 마라.

### 14.4 REPORT.md의 필수 제한 문단

반드시 포함:

- Synthetic ideal-point oracle은 q에 정의되어 있음.
- Simulator 실행 자체가 statistical curves를 바꾸지 않음.
- Short clips이며 RL policy learning/return 비교가 아님.
- Command reconstruction이 human perceptual identifiability나 trajectory-feature geometry를 보장하지 않음.
- Five finite interactions/T=128로 asymptotic limit/rate를 입증하지 않음.
- Correctly specified beta sweep이며 misspecification 검증이 아님.
- SG schedule은 이 측정 구간에서 일정함.
- 기존 prefix 재사용과 numerical approximation을 명시함.

모든 setting에서 RC가 이겨야 한다는 조건은 없다. 실제 결과와 baseline의 transient advantage도 보존하라.

---

## 15. 자원 사용과 실패 처리

사용 가능하고 허용된 CPU/GPU만 적극 활용한다. 다른 사람의 process를 종료하거나 GPU reset, driver/root 설정 변경, overclock, 유료 cloud 사용, 외부 서비스 전송을 하지 마라. 기존 working CUDA/PyTorch와 isolated environment를 우선한다.

OOM이면 checkpoint 후 **computation batch size**를 줄인다. Particles/seeds/M/T를 몰래 줄이지 마라. 반복 OOM을 무한 재시도하지 마라. CPU affinity와 cgroup 제한을 지킨다.

CUDA가 깨져 있으면 실제 오류를 진단하고, 검증된 CPU backend로 처리 가능하면 측정 후 진행한다. GPU가 있다고 가정한 채 fake success를 만들지 마라.

Numerical reference refinement에도 합리적인 시도 한도를 두고 unresolved를 보고한다. Blocker가 있으면 완료한 부분, 구체적인 오류, 필요한 최소 개입을 알려라. 코드나 결과를 존재하는 것처럼 지어내지 마라.

---

## 16. 완료 조건

다음이 모두 충족되거나 미완료 이유가 명시되어야 한다.

- 7 settings × 6 methods × 5 interactions에 각각 actual answers 128개.
- Statistical/numerical tests 통과 또는 unresolved 항목의 명확한 표시.
- 모든 selected candidate에 request와 executed/cache-hit/failed status 존재.
- 실제 pinned Meta-World에서 실행한 numeric trajectories.
- Command/actual-motion diagnostics와 실패 기록.
- 네 statistical figures, endpoint tables, robot-audit outputs.
- Reuse/provenance, seeds, versions, hardware/runtime, limitations 기록.
- Resume 동작 확인, 다른 작업 보존.
- `REPORT.md`에서 새 계산·재사용·replay·검증·입증하지 못한 것을 구분.

**실행 순서:** 기존 코드/결과 확인 → correctness smoke test → robot adapter engineering calibration → 짧은 hardware profiling → manifest freeze → full inference와 rollout 동시 실행 → fixed-history numerical validation → figures/report.

이 작업을 계획서 작성에서 끝내지 마라. 다만 실행 권한/환경이 실제로 없을 때 실행했다고 주장하지 마라.

---

## 17. 최종 설정 YAML

다음 블록으로 `metaworld_fast/config/final.yaml`을 생성하라. 실제 path와 profiling으로 정하는 resource values 외에 statistical settings를 임의로 변경하지 마라.

```yaml
experiment:
  name: rcmlq_metaworld_realization_t128_s5
  dimension: 4
  answers_per_run: 128
  interaction_ids: [0, 1, 2, 3, 4]
  root_seed: 20261004
  methods: [RC, PQ, QI, PS, GC, SG]
  prior: uniform_hypercube
  prior_low: -1.0
  prior_high: 1.0
  oracle: ideal_point_squared_distance_mnl
  inference_beta_matches_oracle: true
  theta_shared_across_methods: true
  theta_shared_across_settings: false
  settings:
    - {id: m02_b04, choices: 2, beta: 4.0}
    - {id: m04_b04, choices: 4, beta: 4.0}
    - {id: m08_b04, choices: 8, beta: 4.0}
    - {id: m16_b04, choices: 16, beta: 4.0}
    - {id: m04_b02, choices: 4, beta: 2.0}
    - {id: m04_b08, choices: 4, beta: 8.0}
    - {id: m04_b16, choices: 4, beta: 16.0}

queries:
  scale_grid: [0.125, 0.2, 0.32, 0.5, 0.8, 1.25, 2.0]
  gaussian_cap_radius: 32.0
  gaussian_cap_center: current_posterior_mean
  clip_to_prior_domain: false
  rc_project_selected_query: false
  fisher_reference: projected_median_center
  scale_tie_rule: smallest_maximizer
  direction_tie_rule: smallest_coordinate_top_eigenspace_projection
  qi_inner_transformations: 1
  ps_with_replacement: true
  gc_covariance_multiplier: 1.0
  sg_stage_length_parameter: 128
  sg_stage_formula: floor_log2_one_plus_t_over_128
  sg_std_formula: two_to_negative_stage
  sg_center: posterior_mean_saved_at_stage_entry

inference:
  reference_dtype: float64
  allow_unvalidated_mixed_precision: false
  marginal_methods: [RC, PQ, QI]
  joint_methods: [PS, GC, SG]
  marginal_initial_nodes: 257
  marginal_refinement_nodes: 513
  marginal_adaptive_mesh: true
  marginal_replay_history_on_remesh: true
  smc_particles: 4096
  smc_reference_particles: 8192
  smc_tempering_ess_fraction: 0.8
  smc_final_resampling_ess_fraction: 0.5
  smc_metropolis_sweeps_per_resampling: 10
  smc_exact_full_history_target: true
  smc_delayed_acceptance_exact_correction: true
  reject_particles_outside_prior: true

measurements:
  primary: [realized_information_ratio, posterior_covariance_trace]
  save_realized_parameter_squared_error: true
  eig_evaluator: integrated_posterior_kl
  eig_reference: same_method_same_posterior_seven_scale_max
  clip_information_ratio_at_one: false
  add_denominator_floor: false
  record_all_preanswer_rounds: true
  record_risk_at_final_posterior: true
  random_query_diagnostic_methods: [PS, GC, SG]
  random_query_diagnostic_t: [0, 31, 63, 95, 127]
  random_query_diagnostic_draws: 64

robot_audit:
  enabled: true
  interpretation: coordinate_model_with_simulator_realization_audit
  environment: reach-v3
  environment_api: inspect_installed_official_metaworld
  clip_action_steps: 50
  terminate_on_success: false
  context_shared_across_methods_and_settings: true
  context_count: 5
  use_scripted_expert_for_nominal_only: true
  train_rl_policy: false
  nearest_neighbor_retrieval: false
  per_query_trajectory_optimization: false
  adapter: four_cartesian_temporal_bases_gripper_fixed
  basis: [sin_pi_x, sin_pi_y, sin_pi_z, sin_2pi_x]
  baseline_cartesian_action_scale: 0.6
  conservative_query_coordinate_bound: 33.0
  perturbation_component_budget: 0.35
  global_gain_formula: perturbation_budget_divided_by_q_bound_and_max_basis_row_l1
  per_query_rescaling: false
  allow_silent_action_clipping: false
  reconstruct_query_from_submitted_actions: true
  initial_command_reconstruction_tolerance: 0.0001
  audit_executed_motion_rank: true
  audit_workspace_saturation: true
  pixel_rendering_in_hot_path: false
  selected_query_rollout_coverage: all
  asynchronous_replay: true
  reference_rollout_audit_interaction_ids: [0]
  reference_rollout_audit_t: [0, 127]
  replay_all_random_diagnostic_queries: false

validation:
  marginal_relative_variance_eig_tolerance: 0.005
  absolute_information_ratio_tolerance: 0.005
  frozen_history_posterior_check_answer_counts: [0, 32, 64, 128]
  small_case_dimension: 2
  small_case_answers: 32
  small_case_choices: [2, 3, 4]
  independent_reference_replays: true
  refine_unresolved_references: true
  report_integration_error_separately: true

resources:
  detect_actual_hardware: true
  optimize_for: validated_end_to_end_wall_time
  gpu_processes: 1
  gpu_batch_size: auto_profile
  gpu_memory_reserve_fraction: 0.15
  cpu_rollout_workers: auto_profile
  cpu_worker_candidates: [4, 8, 12, 16, 20, 24]
  rollout_worker_blas_threads: 1
  rollout_chunk_candidates: auto_profile
  worker_reuse: true
  import_cuda_in_rollout_workers: false
  multiprocessing_start_method: spawn
  reuse_compatible_statistical_prefixes: true
  exact_rollout_cache: true
  approximate_query_cache: false
  overwrite_unrelated_outputs: false
  reduce_statistical_budget_on_oom: false

output:
  root: metaworld_fast/outputs
  checkpoint_every_answers: 16
  atomic_writes: true
  resume: true
  save_full_query_answer_history: true
  save_particle_checkpoints: true
  save_executed_end_effector_series: true
  save_all_pixels: false
  confidence_display: individual_five_interactions_and_mean
  automatic_p_values: false
  automatic_bootstrap_95ci: false
  stop_or_expand_based_on_method_rankings: false
  qualitative_setting: m04_b04
  qualitative_interaction_ids: [0]
  qualitative_methods: [RC, PQ, SG]
  qualitative_t: [0, 63, 127]
  plot_formats: [pdf, png]
```

---

## 18. 원고 근거 및 공식 engineering 참고자료

Selector의 기준은 다른 논문의 이름이 같은 구현이 아니라 첨부 원고의 continuous specialization이다.

- **[P1]** `Ideal_Type_Finder(3).pdf`, PDF p.3, Section 3.1, (1)–(5): fixed ideal point, squared-distance logit, Bayesian update.
- **[P2]** PDF pp.3–6, Section 3.2/4, (6)–(10), (26)–(37), Algorithm 1: moments, projection/quantiles, scale/Fisher.
- **[P3]** PDF pp.6–7, (47)–(58): same-posterior reference, realized/conditional ratio, Bayes risk와 theorem 범위.
- **[P4]** PDF pp.35–36, Appendix A.10: particle update, numerical EIG, scoring cost와 finite-particle 제한.
- **[P5]** PDF p.37, Appendix B: original settings, seven-scale grid, cap, 여섯 selector, one-step QI, SG schedule.
- **[P6]** PDF p.38, Appendix B: 257-node marginal integration, 4096-particle joint SMC, numerical checks, 측정값/diagnostics.
- **[P7]** PDF pp.16–17, Appendix A.2.4: fixed projection과 discrete catalog retrieval의 차이.

공식 문서는 준비 시 확인했지만 실행 시에는 설치 버전의 API와 대조하라.

- **[D1]** Meta-World action space: `https://metaworld.farama.org/benchmark/action_space/`
- **[D2]** Scripted experts: `https://metaworld.farama.org/benchmark/expert_trajectories/`
- **[D3]** Environment 생성 및 rendering/vector options: `https://metaworld.farama.org/introduction/basic_usage/`
- **[D4]** PyTorch multiprocessing: `https://docs.pytorch.org/docs/stable/notes/multiprocessing.html`
- **[D5]** MuJoCo Python/rollout semantics: `https://mujoco.readthedocs.io/en/stable/python.html`

네 개 motion basis, gain bound, 작업 개수, 안정화 KL 식, caching/병렬 구조, T=128에서 SG가 constant stage라는 판단은 이 명세에서 명시하거나 식으로 도출한 내용이다. 외부 robotics benchmark가 이미 검증한 성능 결과처럼 인용하지 마라.

# HARD REQUIREMENT: MAXIMIZE WALL-CLOCK SPEED ON ACSS 4090 SERVER

Run the final experiment on `acss-4090-server`.

Available hardware:
- NVIDIA RTX 4090 x4
- ONLY GPUs 0, 1, and 2 may be used.
- GPU 3 MUST remain completely unused.
- AMD EPYC 9354: 32 physical cores / 64 logical CPUs
- 251 GiB system RAM

The primary optimization objective is MINIMUM WALL-CLOCK COMPLETION TIME.
Do not optimize for simplicity of implementation if a safe parallel implementation
can finish substantially faster.

## 1. Parallelize independent experiments, NOT one experiment across GPUs

Do NOT use DDP, DataParallel, or multi-GPU synchronization for a single run.

Define one independent job as:

    (M, beta, seed, method)

The final required experiment contains:
- d = 4
- T = 128
- 5 independent seeds
- methods = {RC, PQ, QI, PS, GC, SG}
- M sweep:
    M ∈ {2, 4, 8, 16}, beta = 4
- beta sweep:
    beta ∈ {2, 4, 8, 16}, M = 4
- (M=4, beta=4) is shared and MUST be executed only once.

Create a single dynamic global job queue.

Run THREE independent persistent GPU workers:
- worker 0 -> CUDA device 0
- worker 1 -> CUDA device 1
- worker 2 -> CUDA device 2

GPU 3 MUST NEVER be touched.

As soon as a GPU worker finishes one job, it must immediately pull the next
unfinished eligible job from the queue. Do not statically assign an equal number
of jobs if this causes GPUs to become idle.

## 2. Prioritize GPU-heavy methods

The most computationally expensive methods are expected to be:

    PS
    GC
    SG

because they use joint-posterior particle inference.

Schedule these jobs preferentially onto GPUs 0, 1, and 2.

Vectorize on GPU as aggressively as possible:
- likelihood evaluation over all particles
- M-way candidate probabilities
- posterior weighted moments
- covariance computation
- integrated EIG
- evaluation of all seven reference scales
- Gaussian candidate generation
- particle weight updates
- ESS computation
- batched Metropolis proposal evaluation
- any full-history likelihood computation that can be expressed as tensors

Do not execute particle loops in Python when they can be represented as one
batched PyTorch/CUDA operation.

Use one persistent CUDA process per physical GPU rather than repeatedly launching
new CUDA processes for every job.

Keep reusable tensors and posterior state on the GPU whenever practical.

Use torch.no_grad() / inference_mode wherever gradients are unnecessary.

Do not move tensors repeatedly CPU <-> GPU inside an inner round loop.

## 3. Use the CPU concurrently

While GPUs 0-2 execute GPU-heavy jobs, the CPU must simultaneously execute:
- Meta-World / MuJoCo rollouts
- RC jobs
- PQ jobs
- QI jobs
- preprocessing
- trajectory realization
- logging/checkpoint serialization when it does not block GPU execution

Use persistent multiprocessing workers.

Start benchmarking with 24 CPU worker processes.

Before the full run, benchmark:
    16 workers
    24 workers
    32 workers

on a short representative workload and automatically choose the configuration
with the highest stable throughput.

Do NOT blindly use all 64 logical threads if it decreases throughput.

For every CPU worker set:

    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1

and also set the equivalent PyTorch CPU thread count to 1 inside multiprocessing
workers.

This is required to prevent nested BLAS/OpenMP oversubscription.

## 4. Meta-World execution must be optimized for throughput

Run Meta-World/MuJoCo HEADLESS.

During the full numerical experiment:
- DO NOT render RGB images
- DO NOT record video
- DO NOT open viewers
- DO NOT create a new environment for every candidate
- DO NOT repeatedly reload assets

Each CPU worker should create a persistent environment once and reuse it.

Reset the environment state as needed between trajectories rather than recreating
the simulator.

Batch / asynchronously dispatch candidate trajectory realization whenever
possible.

Only after the complete numerical experiment finishes may a very small number of
representative trajectories be rendered for visualization.

## 5. Preserve experimental pairing under asynchronous execution

For every unique:

    (M, beta, seed)

generate theta_star deterministically BEFORE launching method jobs.

Save it to disk.

All six methods for that setting and seed MUST use exactly the same theta_star,
even if those jobs run:
- on different GPUs,
- on different CPUs,
- at different times,
- or after a restart.

Do not allow scheduling order to affect any random seed.

Derive all RNG seeds deterministically from:
    master_seed
    setting
    interaction_seed
    method
    purpose

and record them.

## 6. Do not recompute shared quantities unnecessarily

Reuse the unique (M=4, beta=4) experiment between the M sweep and beta sweep.

Cache deterministic quantities that depend only on:
- M
- beta
- the fixed scale grid
- fixed trajectory basis / Meta-World realization settings

Do not repeatedly recompute static constants.

Within one posterior state, evaluate all seven reference scales in ONE batched
operation whenever possible.

Within an M-way query, evaluate all M candidates together rather than looping over
candidates in Python.

## 7. Pipeline CPU and GPU work asynchronously

Do not force the GPU to wait for Meta-World unnecessarily.

Implement producer/consumer queues where useful:

    CPU rollout / preparation
             |
             v
        GPU posterior/EIG
             |
             v
       next queued work

Likewise, CPU workers should continue processing while GPU workers execute SMC.

If a synchronization barrier is not mathematically necessary, remove it.

## 8. Checkpoint and resume without adding large overhead

Every independent job must be resumable.

Maintain an atomic manifest containing:

    pending
    running
    completed
    failed

for every (M, beta, seed, method) job.

Checkpoint posterior state and experiment state periodically, e.g. every
8-16 rounds, and at job completion.

Use atomic temporary-file -> rename writes.

On restart:
- detect completed jobs and skip them
- resume partially completed jobs from their latest valid checkpoint
- never redo a completed 128-round interaction

Do not checkpoint enormous redundant tensors if they can be reconstructed cheaply.

## 9. Perform an actual throughput benchmark BEFORE committing to the full run

Before launching all final jobs, run approximately 5-10 minutes of representative
work containing at least:

    RC
    PS
    GC
    SG

and at least:
    M = 4
    M = 16

Measure and print:

- rounds / second by method
- candidate trajectories / second
- seconds per complete 128-round interaction
- GPU utilization on devices 0, 1, 2
- GPU memory usage
- CPU utilization
- active CPU worker count
- queue wait time
- estimated remaining wall-clock time for the entire final experiment

Use these measurements to tune:
- CPU worker count
- GPU batch size
- number of concurrently prepared rollouts
- particle evaluation batch sizes

Do this automatically before the full production run.

## 10. Detect bottlenecks and react

If GPU utilization is consistently below ~70% while CPU resources are available:
- increase GPU batching,
- increase asynchronous CPU preparation,
- remove unnecessary synchronization,
- batch scale/candidate/particle evaluation more aggressively.

If CPU is saturated and GPU workers wait:
- reduce simulator overhead,
- reuse persistent environments,
- reduce Python serialization,
- tune the CPU worker count,
- move tensorizable calculations to CUDA.

If RAM or VRAM usage approaches unsafe levels:
- reduce only batch concurrency,
- DO NOT reduce T, seeds, methods, particles, settings, or required numerical
  accuracy.

Never silently reduce the experiment to make it finish faster.

## 11. GPU safety

Explicitly enforce:

    CUDA_VISIBLE_DEVICES=0,1,2

or equivalent device isolation.

Print the physical GPU mapping at startup.

Abort with an error if the experiment detects or attempts to allocate GPU 3.

## 12. Final execution requirement

After benchmarking and optimization, RUN THE FULL EXPERIMENT.

Do not stop after:
- writing code,
- producing a benchmark,
- producing a TODO,
- or describing how it should be run.

Execute all required jobs, resume failures, verify completion, aggregate results,
generate the final plots/tables, and write the final REPORT.md.

The final completion check must confirm:

    7 unique (M, beta) settings
    x 5 seeds
    x 6 methods
    x 128 rounds

with no missing jobs.

Do not modify the scientific definitions of RC, PQ, QI, PS, GC, SG,
the response likelihood, information ratio, Bayes MSE, scale grid, posterior
update, or Meta-World query-realization protocol merely to obtain faster runtime.
Only implementation-level optimization and parallelism are allowed.