# 실행 가능한 재구현 명세: real-response GLM의 15개 acquisition algorithm

이 문서는 Codex 또는 다른 구현자가 현재 benchmark의 알고리즘을 다시 구현할 때 필요한 입력/출력, 수식, 순서, 수치 처리와 RNG 규약을 제공한다. 9개 공통 방법과 family별 3개씩, 총 15개 고유 알고리즘을 다룬다. 한 family의 figure에는 이 중 12개가 등장한다. source snapshot은 아래 명세의 실행 증거이며, 논문 구현과 다른 adaptation을 구별하여 적었다.

## A. 공통 인터페이스와 금지 의존성

함수의 권장 입력은 다음과 같다.

```
select_batch(method, X_pool[N,d], observed_ids[n], candidate_ids[K],
             theta_hat[d], family, m, rho, rng, dataset, seed, round_index,
             observed_y=None, planned_rounds=None, settings={})
    -> selected_global_ids[m], audit
```

float array는 모두 float64, global row ID는 int64이다. `X_pool[:, :d]`는 기존 고정 query이다. intercept가 이미 첫 열에 있고 d에 포함되므로 새 열을 붙이지 않는다. features를 method별로 scaling/whitening하지 않는다. candidate ID는 unique이며 observed IDs와 disjoint여야 한다. `0 < m <= K`를 검사한다. 반환 ID는 candidate의 원래 global row ID이며 정확히 m개, 중복 없음이어야 한다.

`observed_y`는 B-EMCM에만 전달한다. 그 shape는 `(n,)`이고 observed_ids와 같은 순서이다. 전체 y_pool, θ_ref, V_ref, evaluation metric, 다른 알고리즘 결과는 selector 인자로 넘기지 않는다. runner가 selector 반환 후에만 신규 label을 공개하고 GLM을 다시 fit한다. θ_hat은 batch가 끝날 때까지 고정이다. 모델 curvature도 batch 시작 θ에서 고정한다.

Gaussian ridge fit: `(X.T X + rho I) theta = X.T y`를 Cholesky로 푼다. Bernoulli는 sum logistic loss + rho*||theta||²/2의 gradient/Hessian을 사용한다. rho=1, intercept도 penalize한다. mean loss + fixed ridge로 바꾸지 않는다. fitting tolerances와 reference 설명은 EXPERIMENT_PROTOCOL.md 4–5절에 있으며 `source/real_response/numerics.py`가 실제 구현이다.

## B. Candidate, tie, RNG를 정확히 재현하기

초기 80개 ID는 제공된 `data/initial`을 읽는다. 새로 생성할 때는 최대 d=20, label-free greedy D-opt, hash 후보 2,000개를 사용한다. 이후 round의 후보는 아직 관측하지 않은 ID에 다음 priority를 주고 작은 순으로 K개까지 취한다. K=256 Gaussian, 512 Bernoulli.

```python
salt = int.from_bytes(sha256(repr(key).encode()).digest()[:8], 'little')
z = uint64(ids) ^ uint64(salt)
z += uint64(0x9E3779B97F4A7C15)
z = (z ^ (z >> 30)) * uint64(0xBF58476D1CE4E5B9)
z = (z ^ (z >> 27)) * uint64(0x94D049BB133111EB)
priority = z ^ (z >> 31)  # unsigned 64-bit wraparound is intentional
```

후보 key는 `('candidate', dataset, int(seed), int(round_index))`, tie key는 `('tie', dataset, int(seed), int(round_index))`, 초기 key는 `('initialization', dataset, int(seed))`이다. SHA 입력은 Python tuple repr이며 `'|'` 연결이 아니다.

stochastic selector seed는 별도 규칙이다.

```python
text = '|'.join(map(str, ('selector', dataset, d, seed, method)))
selector_seed = int.from_bytes(sha256(text.encode()).digest()[:8], 'little')
rng = numpy.random.default_rng(selector_seed)
```

trajectory 전체에서 같은 rng를 이어 쓴다. 매 round마다 다시 seed하지 않는다. batch 전후 bit_generator.state를 audit에 저장한다.

일반적인 max-score tie는 유효 후보 중 best를 찾고 `abs(score-best) <= 1e-11*max(1,abs(best))`인 후보 중 tie priority가 가장 작은 것으로 푼다. min-score이면 best를 min으로 바꾼다. 모든 유효 score의 finite 여부를 먼저 검사한다. Entropy, TS, SDP의 top-weight 정렬은 `np.lexsort((priority, -score))`의 exact ordering을 사용한다. 구현상 이 구분을 지킨다.

## C. 공통 Fisher primitive

`C = X_pool[candidate_ids]`, `X0 = X_pool[observed_ids]`로 둔다. exact curvature는

\[
w(x)=1\quad(Gaussian),\qquad w(x)=\sigma(x^T\theta)\sigma(-x^T\theta)\quad(Bernoulli).
\]

공통 Fisher-based 방법에서 acquisition floor ε는 Gaussian 0, Bernoulli 1e−12이며

\[
\widetilde w_i=\max(w_i,\epsilon),\quad u_i=\sqrt{\widetilde w_i}x_i,
\quad W_0=\rho I+\sum_{x\in X_0}\max(w(x),\epsilon)xx^T.
\]

W는 매 round 시작에 현재 θ로 다시 계산한다. 과거 θ로 계산한 Fisher를 영구 누적하면 안 된다. slot마다 고른 u_j의 `u_j u_j.T`를 W에 더하되 θ를 refit하지 않는다. `U`는 K×d 행렬이다. inverse/solve는 Cholesky를 우선하고 대칭성을 복원한다. ridge 때문에 이 setting에서 W는 SPD이다.

rank-one update는 A=W^{-1}, v=A u_j, q=u_j^T v일 때

\[
(W+u_ju_j^T)^{-1}=A-vv^T/(1+q).
\]

downdate는 `A + vv.T/(1-q)`이다. near-zero denominator를 무시하지 않는다. 기존 cache는 32회 update마다 direct inverse와 비교하고 relative matrix error >1e−9이면 rebuild한다. 안정성용 rebuild는 objective를 바꾸지 않는다. 이 cache를 사용하지 않고 직접 solve하는 재구현도 수학적으로 동일하나 거의 동률인 후보의 선택은 numerical tolerance에 민감할 수 있다. bit-for-bit trajectory 재현이 필요하면 source의 operation order를 유지한다.

평가용 V_ref/V_hat에는 ε floor를 쓰지 않는다. 공통 acquisition의 W와 metric의 V_hat을 혼동하지 않는다.

## 1. ours — FB-opt2

내부 method ID는 `fb_opt2`, 버전은 `fb_opt2_v1_range_anchored`이다. 다음 계획 예산을 먼저 고정한다.

\[
T=N_{\max}/m,\quad N_{\max}=400\ (Gaussian),\ 4000\ (Bernoulli),
\quad R=1\ (Gaussian),\ 1/4\ (Bernoulli).
\]
\[
\tau_{\rm range}=\max(10^{-8}, R\sqrt{m/(8\log(\max(2,d)))}),\quad
\tau_0=\tau_{\rm range}N_{\max}^{-3/4},\quad
\tau_t=\tau_{\rm range}(mt/N_{\max})^{3/4}.
\]

multiplier=1, t=1,…,T이다. initial 80개는 mt에 포함하지 않는다. 한 batch의 모든 slot에 같은 τ_t를 사용한다.

slot마다 virtual W의 eigendecomposition을 `W=Q diag(lambda) Q.T`로 구하고

\[
p_k=\frac{\exp[-(\lambda_k-\lambda_{\min})/\tau_t]}{\sum_l\exp[-(\lambda_l-\lambda_{\min})/\tau_t]},
\quad s_i=u_i^T Q\operatorname{diag}(p)Q^Tu_i=\sum_k p_k(u_i^Tq_k)^2.
\]

유효 후보 중 s_i 최대를 선택하고 W←W+u_i u_i^T, 해당 candidate를 mask한다. m회 반복한다. finite τ를 단순 최저 eigenvector 선택으로 대체하지 않는다. 이 값은 Gibbs density에 대한 rank-one Fisher score이며 exact finite-step λ_min gain과 다르다. 그 차이가 E-greedy와의 구별이다.

Audit: fixed horizon, Nmax, range_R, tau_range, tau0_opt2, 현재 τ, 각 slot의 score/tie/ID, batch 시작 W/θ/curvature. 마지막 batch에서 τ_t=tau_range가 성립해야 한다. source: selectors.py의 `fb_opt2_scale`, `fb_opt2_temperature`, `select_batch`.

## 2. A-opt

목표는 virtual `trace(W^{-1})`를 줄이는 것이다. slot score는 Sherman–Morrison으로 구한 정확한 감소량

\[
s_i=\frac{u_i^T W^{-2}u_i}{1+u_i^TW^{-1}u_i}.
\]

s_i 최대 후보를 고르고 virtual rank-one update한다. m회 반복한다. numerator에 target Fisher F를 끼우면 BAIT로 바뀌므로 넣지 않는다. matrix inverse를 diagonal approximation으로 바꾸지 않는다. α 등 추가 hyperparameter는 없다.

## 3. D-opt

목표는 virtual log determinant 증가이다.

\[
s_i=\log(1+u_i^TW^{-1}u_i).
\]

`log1p`로 안정적으로 평가하고 최대 후보 선택 후 virtual update한다. m회 반복한다. 레버리지에 비례한 확률 sampling이 아니라 deterministic greedy이다. θ_ref가 아닌 현재 θ_hat으로 curvature를 계산한다.

## 4. E-greedy

slot score는

\[
s_i=\lambda_{\min}(W+u_i u_i^T).
\]

최대 후보를 선택한 뒤 W를 update한다. dense eigvalsh로 모든 후보를 평가하면 정확한 기본 구현이 된다. 기존 실행은 d=5에 batched dense eigensolver, d=10,20에 rank-one secular equation solver를 사용한다. 후자는 λ₁과 λ₂ 사이 최저 root를 찾으며 repeated minimum eigenvalue 및 zero coupling의 deflation을 처리한다. secular rtol=1e−12, 최대 48회이고 final winner 주변 상위 4개 후보를 dense eigvalsh로 확인한다. score 차이가 허용량을 넘으면 모든 유효 후보를 dense로 재계산한다. source의 `e_scores_secular`, `_e_greedy`가 정확한 numerical guard를 제공한다.

E-greedy를 `u_i.T v_min`의 제곱 최대화만으로 구현하면 정확한 λ_min after-update objective와 다르다. τ=0 FisherBalancing과 동일하다고 가정하지 않는다.

## 5. E-SDP + rounding (표시 E-SDP)

이 benchmark에서 E-SDP는 capped approximate design SDP와 명시적 rounding heuristic이다. globally optimal integer E-design이나 Allen-Zhu 논문의 sparsifier 전체 구현이라고 부르지 않는다.

\[
\max_{a,z}z\quad\text{s.t.}\quad
W_0+\sum_i a_i u_i u_i^T-zI\succeq0,\quad
0\le a_i\le1,\quad\sum_i a_i=m.
\]

### Continuous solve

scale=`max(1,trace(W0)/d,max_i ||u_i||²)`로 W0와 candidate rank-one terms를 나눈 동일 SDP를 푼다. CVXPY DPP template를 (d,K,solver)별로 cache한다. candidate가 바뀌므로 이전 a값을 신뢰하는 numerical warm start는 하지 않는다. 초기 a=m/K이다.

시도 순서는 SCS(5000 iterations, eps_abs=eps_rel=1e−4), strict SCS(20000,5e−6), CLARABEL(1000,tol_gap_abs=tol_gap_rel=tol_feas=1e−7)이다. 이전 시도가 아래 external gate를 통과하면 중단한다. `optimal_inaccurate`를 자동 수락하지 않는다.

- solver status가 정확히 `optimal`.
- raw a의 equality error ≤5e−4*max(1,m), bound violation ≤5e−4.
- `max(0,z−lambda_min(W+Σa_i u_i u_i.T))/scale ≤5e−4`.
- 아래 valid upper bound UB와 projected fractional feasible objective LB_frac의 `(UB-LB_frac)/max(1,abs(UB),abs(LB_frac))`가 [−1e−9,2e−3] 범위.

raw a를 `clip(a−c,0,1)`로 sum=m이 되도록 80회 bisection해 capped simplex에 projection한다. sum error >1e−9이면 실패한다.

SDP PSD dual을 대칭화하여 negative eigenvalues를 0으로 projection하고 trace=1로 normalize한 P를 만든다. 유효하지 않거나 positive trace ≤1e−15이면 P=I/d이다. 모든 P≽0,trP=1에 대해

\[
UB=\operatorname{tr}(PW_0)+\text{sum of largest m of }\{u_i^TPu_i\}
\]

가 valid bound이다. dual projection은 bound를 만들기 위한 것이며 metric eigenvalue clipping이 아니다.

### Rounding와 swap

1. projected a가 큰 top-m 집합 1개, dependent rounding 8개, E-greedy 집합 1개를 만든다.
2. dependent rounding에서는 fractional entry를 candidate 순서상 처음 2개씩 잡는다. α=min(1-p_i,p_j), β=min(p_i,1-p_j). 확률 β/(α+β)로 `(p_i,p_j)+=(α,−α)`, 아니면 `(−β,β)`를 적용한다. 모든 항이 0/1이 될 때까지 반복한다. 정확한 m개 cardinality를 확인한다.
3. 10개 집합 중 실제 `lambda_min(W0+Σselected u_i u_i.T)`가 최대인 것을 고른다. exact tie는 먼저 나온 집합을 고른다.
4. 현재 집합에서 a가 작은 8개, 외부에서 a가 큰 16개를 대상으로 최대 128개 one-swap 후보를 평가한다. 현재 집합 기준 한 pass의 best swap 하나만 적용한다. 개선 허용량은 `1e−11*max(1,abs(best_value))`이다.
5. 최종 objective가 E-greedy보다 나쁘지 않고 UB보다 크지 않은지 수치 tolerance로 확인한다.

모든 solver 시도가 실패하면 미리 계산한 E-greedy batch를 반환하고 fallback을 명시한다. 이번에 재사용한 E-SDP에는 두 번의 fallback이 있으며 protocol 문서에 위치가 있다. solver status, residual, raw/projected weights, rounding candidate와 objective, swap, UB/LB/gap, fallback reason을 audit에 남긴다.

## 6. BAIT

이 구현은 frozen current GLM Fisher의 forward–backward BAIT adaptation이다. target은 capped candidates가 아니라 **전체 고정 pool feature**에서

\[
F=\frac1{N_{\rm pool}}\sum_{x\in X_{\rm pool}}w(x)xx^T
\]

로 계산한다. target curvature에는 floor를 적용하지 않는다. selected-set information은 위 W0처럼 sum scale이다. 실제 구현의 이름은 `bait_normalization='matched_sum_information'`이다. neural-network last-layer loss scaling을 복사해 W와 F의 relative scale을 바꾸지 않는다.

W=W0에서 forward phase를 min(2m,K)회 수행한다.

\[
s_i^{+}=\frac{u_i^TW^{-1}FW^{-1}u_i}{1+u_i^TW^{-1}u_i}.
\]

최대 감소량의 후보를 추가하고 virtual W를 update한다. provisional set의 크기가 m을 넘는 동안 backward phase를 수행한다.

\[
s_i^{-}=\frac{u_i^TW^{-1}FW^{-1}u_i}{1-u_i^TW^{-1}u_i}.
\]

현재 포함된 후보 중 제거에 따른 증가량 s_i^-가 가장 작은 것을 제거한다. denominator≤1e−10이면 해당 downdate의 실제 `trace((W-u_i u_i.T)^{-1}F)-trace(W^{-1}F)`를 direct solve로 계산한다. 결과의 순서는 최초 forward 선택 순서 중 살아남은 ID 순서를 유지한다.

Audit에 F, provisional/removed/final IDs, exact final FIR objective, cache rebuild/error를 저장한다. 공식 BAIT 논문 및 공개 코드의 forward–backward Fisher design 원리를 linear GLM로 사용하며, 공식 neural benchmark 전체를 재현했다고 주장하지 않는다.

Reference: [BAIT paper](https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf), [author code](https://raw.githubusercontent.com/JordanAsh/badge/master/query_strategies/bait_sampling.py).

## 7. UCB

alpha=1, frozen natural predictor η_i=x_i^Tθ_hat이다. slot마다

\[
s_i=x_i^T\widehat\theta+\alpha\sqrt{x_i^TW^{-1}x_i}
\]

최대 후보를 고른다. uncertainty numerator에는 u_i가 아니라 원래 x_i를 사용한다. 선택 후 W에는 u_i u_i.T를 더한다. 작은 음수 roundoff는 0으로 자르되 variance <−1e−9이면 실패한다. sigmoid η를 이용한 entropy sampling으로 바꾸지 않는다.

현재 저장된 UCB는 reward/natural-score exploration baseline이다. parameter-estimation risk를 직접 최소화하는 방법으로 해석하지 않는다. 표시 이름 축약 외에 α, objective, update를 바꾸지 않았다.

## 8. TS

alpha=1, W0=L L.T인 lower Cholesky factor를 구한다. **batch당 한 번만** z∼N(0,I_d)를 뽑아

\[
\widetilde\theta=\widehat\theta+\alpha L^{-T}z
\]

로 만든다. 모든 candidate의 `x_i.T theta_tilde`를 계산하여 top-m을 한꺼번에 선택한다. slot마다 sample을 새로 뽑거나 virtual posterior를 다시 fit하지 않는다. precision은 W0, covariance는 alpha² W0^{-1}이다. `L^{-1}z`와 `L^{-T}z`를 바꾸어 구현하면 틀린 covariance가 된다.

이는 현재 ridge/Laplace Gaussian perturbation convention이며 exact Bayesian posterior sampler라고 주장하지 않는다. z와 θ_tilde, rng state를 audit한다.

## 9. Random

현재 capped unqueried 후보의 position에서 `rng.choice(K,size=m,replace=False)`를 사용한다. 전체 미관측 pool에서 직접 m개를 뽑는 방법으로 바꾸지 않는다. response와 θ_hat, Fisher를 점수에 사용하지 않는다. stochastic stream은 위에서 정의한 method-specific PCG64 stream이다.

## 10. B-EMCM — Gaussian only

Reference: Wenbin Cai, Muhan Zhang, Ya Zhang, “Batch Mode Active Learning for Regression With Expected Model Change,” IEEE TNNLS 28(7):1668–1681, 2017, DOI 10.1109/TNNLS.2016.2542184. [Author PDF](https://muhanzhang.github.io/papers/TNNLS_2016.pdf). Linear EMCM Eq.11–12, batch correction Eq.19–25, Algorithm 3을 사용한다.

Z=4, virtual SGD step alpha=0.05이며 paper의 실험 설정에서 가져왔다. observed size n에 대해 bootstrap seed 4개를 `rng.integers(0,2**63-1,size=4)`로 뽑는다. 각 seed로 별도 default_rng를 만들고 n개 observed position을 with replacement로 추출한다. 해당 X,y에 같은 sum-ridge(ρ=1) estimator를 적합하여 θ^(z)를 얻는다. hidden current-batch label을 쓰지 않는다.

candidate의 bootstrap prediction은 `y_iz=C_i.T theta_bootstrap[z]`로 batch 동안 고정한다. 각 bootstrap scenario의 virtual parameter를 `v_z=theta_hat`으로 초기화한다. 매 slot:

\[
r_{iz}=x_i^Tv_z-x_i^T\theta^{(z)},\quad
s_i=\frac1Z\sum_z|r_{iz}|\|x_i\|_2.
\]

유효 후보 중 s_i 최대 j를 고른 뒤, 모든 scenario에

\[
v_z\leftarrow v_z-\alpha\,r_{jz}x_j
\]

를 적용한다. 다음 score는 이 변경된 v_z로 계산한다. 이는

\[
x_i^Tv_z=x_i^T\widehat\theta-\alpha\sum_{j\;\mathrm{previously\ selected}}r_{jz}x_i^Tx_j
\]

라는 correlation correction을 포함한다. 처음 계산한 EMCM score의 top-m을 반환하면 B-EMCM이 아니다.

**Adaptation:** 원 논문의 선형 모델을 현재 ridge estimator로 적합한다. virtual gradient는 위의 per-example loss gradient이며 매 virtual step에 ridge penalty gradient를 다시 넣지 않는다. batch 내 알 수 없는 여러 응답의 공동 scenario는 같은 bootstrap member z가 예측한 값을 일관되게 사용한다(`fixed_bootstrap_member_per_scenario`). 이 coupling convention을 명시한다. 가상 후보 선택 순서에 실제 label이나 θ_ref를 섞지 않는다.

Audit: Z, alpha, bootstrap seeds, bootstrap fitted θ와 convergence, observed count, candidate IDs, 선택 score와 scenario gradient, virtual final θ, runtime. correlated candidate toy에서 m>1 결과가 naive top-m과 다름을 검사했다. alpha=0에서는 naive 점수 순서로 돌아가는지도 검사했다.

## 11. LCMD — Gaussian only

Reference: David Holzmüller, Viktor Zaverkin, Johannes Kästner, Ingo Steinwart, “A Framework and Benchmark for Deep Batch Active Learning for Regression,” JMLR 24(164):1–81, 2023, §5.2.8. [Paper](https://www.jmlr.org/papers/volume24/22-0937/22-0937.pdf).

linear feature map은 φ(x)=x, distance는 기존 query 좌표에서 squared Euclidean distance이다. 초기 center는 관측 완료된 **모든** X0 row이고, batch에서 고른 row도 즉시 새 center가 된다. 고정 intercept 열은 distance에서 상쇄되므로 기존 열을 그대로 쓰고 새 intercept를 붙이지 않는다.

1. 각 남은 candidate를 가장 가까운 center에 assign한다. center distance 동률은 먼저 있던 center를 유지한다.
2. center c의 mass는 배정된 남은 candidate들의 `sum ||x−c||²`이다. 단순 point count가 아니다.
3. maximum mass cluster를 찾는다. cluster mass가 1e−11 상대 허용량 내로 동률이면 그 cluster들의 후보를 합쳐 처리한다.
4. 해당 유효 cluster 안에서 nearest-center squared distance가 가장 큰 candidate를 고른다. candidate score tie는 공통 priority를 쓴다.
5. 선택 row를 mask하고 center에 추가한다. 각 candidate에 대해 이전 nearest distance와 새 center distance를 비교해 더 작은 경우만 갱신한다.
6. m번 반복한다.

label, θ_hat, Fisher, θ_ref가 필요 없다. TP-style training-plus-pool center 구성을 low-dimensional linear query에 적용했다. NTK kernel, sketching, posterior transform을 넣지 않았다. farthest-first는 2–3단계의 largest-cluster 선택이 빠져 있으므로 대체 구현이 아니다. audit에는 초기 center 수, cluster mass, 선택 distance, ID, runtime을 기록한다.

## 12. RegVol — Gaussian only

Reference: Michał Dereziński and Manfred K. Warmuth, “Subsampling for Ridge Regression via Regularized Volume Sampling,” AISTATS 2018, PMLR 84:716–725, FastRegVol Algorithm 2. [Paper page](https://proceedings.mlr.press/v84/derezinski18a.html), [author arXiv PDF](https://arxiv.org/pdf/1710.05110).

매 round의 capped candidate matrix C에서 크기 m의 subset을 다음 exact reverse sampler로 뽑는다. λ=rho=1이며 observed information을 초기 matrix에 추가하지 않는다.

```
active = [0,...,K-1]
Z = inverse(C.T @ C + lambda * I)
while len(active) > m:
    position ~ Uniform({0,...,len(active)-1})
    j = active[position]
    v = Z @ C[j]
    h = 1 - C[j].T @ v
    validate h
    if rng.random() < h:
        Z += outer(v,v)/h
        remove j by swapping active[position] with the last active entry
        every 32 accepted removals:
            Z = inverse(C[active].T @ C[active] + lambda*I)
return sorted remaining candidate positions
```

proposal가 uniform이고 acceptance가 h이므로 accepted removal 확률은 현재 active 집합의 h_i에 비례한다. leverage score로 직접 inclusion sampling하는 것이 아니다. determinant를 explicit하게 계산하지 않는다. 모든 inverse는 symmetric float64 Cholesky이고 rank-one update 후 대칭화한다.

h가 [0,1]에서 1e−10보다 크게 벗어나거나 nonfinite이면 해당 job을 실패시킨다. 오직 1e−10 이내 floating error만 clip하고 clip event를 센다. h=0인 제거가 수락되면 오류다. actual new run에서는 clip 0회였다.

**Adaptation:** 고정-budget one-shot sampler를 매 adaptive batch마다 현재 후보에 적용했다. metadata name은 `RegVol_roundwise`, figure name은 `RegVol`이다. response, θ_hat, θ_ref를 selection에 쓰지 않는다. final selected order는 candidate position 정렬이며 label 공개가 batch 이후라 within-batch order는 성능에 영향을 주지 않는다.

정규화된 역방향 transition denominator가 상태에 의존하므로 λ>0의 최종 subset 확률을 일반적으로 단순 `det(C_S.T C_S+λI)`에 비례한다고 주장하지 않는다. test는 작은 문제의 정확한 reverse-transition 분포를 열거하여 Monte Carlo sampling과 비교했다. 전체 horizon에서의 round-wise distribution이 원 논문의 one-shot mT-size distribution과 같다고도 주장하지 않는다.

Audit: lambda, RNG state/seed, proposal count, accepted reverse-removal count, removal positions, accepted h, min/max h, rebuild/clip count, selected IDs, runtime.

## 13. Entropy — Bernoulli only

표준 binary uncertainty sampling이다. batch 시작 θ에서 `p_i=expit(C_i.T theta_hat)`를 고정하고

\[
H_i=-p_i\log p_i-(1-p_i)\log(1-p_i)
\]

를 계산한다. `scipy.special.xlogy`로 0 log 0=0을 안정적으로 처리한다. 확률 자체를 0.5 쪽으로 clipping하지 않는다. entropy top-m을 exact lexsort로 선택한다. virtual refit, Fisher update, diversity term이 없다. 선택 후 한꺼번에 실제 응답을 공개한다.

이 표준 uncertainty rule은 classical active learning literature의 predictive-entropy criterion이며, 특별한 batch-diversity algorithm으로 부르지 않는다. source: family_selectors.py의 `select_family_batch` entropy branch. audit에 p, H, chosen IDs/scores, runtime을 저장한다. 테스트는 p≈0.5 최대, 극단 p≈0/1에서 finite 작은 entropy를 확인한다.

## 14. BADGE — Bernoulli only, 저장 결과 재사용

Reference: Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal, “Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds,” ICLR 2020, Algorithm 1. [Paper](https://arxiv.org/abs/1906.03671), [author code](https://raw.githubusercontent.com/JordanAsh/badge/master/query_strategies/badge_sampling.py).

batch 시작 θ를 고정하여

\[
p_i=\sigma(x_i^T\theta),\quad \widetilde y_i=\mathbf1[p_i\ge1/2],\quad
G_i=(p_i-\widetilde y_i)x_i
\]

를 만든다. binary single-logit GLM의 gradient embedding이다. 실제 intercept 좌표 1/s가 이미 x에 있으므로 gradient에도 포함된다. true label은 쓰지 않는다.

첫 center는 ||G_i||² 최대 candidate를 공통 tie 규칙으로 고른다(논문 BADGE initialization). 이후에는 매 candidate의 이미 선택한 center까지 최소 squared distance D_i²를 유지한다. 남은 후보에 대해 이 값에 비례하여 **하나의 D² draw**로 다음 center를 뽑는다. uniform u를 한 번 뽑고 누적합에 `searchsorted(...,side='right')`를 사용한다. 이미 고른 ID의 mass는 0으로 만든다. 모든 mass가 0이면 남은 ID에서 uniform draw를 수행하고 fallback count를 기록한다.

sklearn의 여러 local trial을 수행하는 greedy k-means++ 변형을 쓰지 않는다. Lloyd iteration도 하지 않는다. 단순 farthest-first로 바꾸지 않는다. m개의 center를 선택하면 종료하고 label을 공개한다.

90개 저장 trajectory를 재사용했으며 새로 fit/select하지 않았다. 기존 batch audit에 후보 ID, frozen θ, RNG state, D² draw가 있고, 이번 패키지의 `audits/reused_badge/*.npz`에는 저장된 pre-batch θ로 사후 계산한 candidate별 gradient-embedding norm을 보강했다. 이는 새 simulation 결과가 아니라 post-hoc diagnostic이다.

## 15. FIRAL — Bernoulli binary specialization

Reference: Youguang Chen and George Biros, “FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression,” NeurIPS 2023. [Paper](https://papers.neurips.cc/paper_files/paper/2023/file/cfcadfe84ee49908cde1fc2992c38d20-Paper-Conference.pdf), [official supplement and author code](https://proceedings.neurips.cc/paper_files/paper/2023/file/cfcadfe84ee49908cde1fc2992c38d20-Supplemental-Conference.zip). Eq.10–15,17–23, Algorithm 1, Appendix G.2를 사용한다.

### 15.1 Binary Fisher와 observed contribution

θ를 고정하고 **exact**, nonfloored curvature w_i=σ(η_i)σ(−η_i)를 구한다. `U_i=sqrt(w_i)*x_i`, `H_i=U_i U_i.T`이다. 단일 binary logit이므로 d차원이며 불필요하게 2d 또는 multiclass parameter를 만들지 않는다.

\[
C_0=\rho I+\sum_{x\in X_0}H(x),\quad F=\frac1K\sum_{i\in\text{current capped candidates}}H_i.
\]

논문의 labeled information을 batch slot에 C0/m씩 분배한 Hbar_i=H_i+C0/m으로 해석한다. fixed ridge ρI를 C0에 포함한 것은 현재 estimator에 맞춘 명시적 prior adaptation이다. F target은 이번 candidate set이며 BAIT의 full-pool target과 다르다. 실제 y를 읽지 않고 이미 적합한 θ만 쓴다.

### 15.2 Continuous capped relaxation

\[
\min_z f(z)=\operatorname{tr}(\Sigma(z)^{-1}F),\quad
\Sigma(z)=C_0+\sum_i z_iH_i,\quad 0\le z_i\le1,\quad \sum_i z_i=m.
\]

A=Σ^{-1}일 때 gradient는 `grad_i = -U_i.T A F A U_i`이다. 현재 구현은 paper의 mirror-descent solver 대신 다음 certified pairwise-coordinate optimizer로 **동일 objective와 사용자가 요청한 capped constraints**를 푼다. optimizer 교체는 FTRL discretization을 greedy rule로 교체하는 것이 아니다.

1. z_i=m/K로 초기화한다. 그 지점 gradient의 최소 m개 coordinate를 1로 한 feasible vertex의 objective가 더 낮으면 해당 vertex에서 시작한다.
2. 현재 gradient의 최소 m개 coordinate를 1로 한 v를 구하고 `gap=grad.T@(z-v)`를 계산한다. convexity로 f(z)−gap는 valid lower bound이다.
3. `gap <= 1e−7*max(abs(f),1e−15)`이면 인증 완료.
4. z_i<1−1e−13 중 gradient 최소인 add index i, z_j>1e−13 중 gradient 최대인 remove index j를 고른다. 최대 step b=min(1−z_i,z_j).
5. `z_i+=gamma, z_j-=gamma`, gamma∈[0,b]의 정확한 1D convex line minimum을 구한다. Δ=H_i−H_j, Σ^{-1/2}ΔΣ^{-1/2}=V diag(e) V.T, `a=diag(V.T Σ^{-1/2} F Σ^{-1/2} V)`이면 derivative는 `-sum(a*e/(1+gamma*e)^2)`이다.
6. derivative(b)≤0이면 gamma=b. 아니면 brentq([0,b],xtol=1e−13)로 derivative root를 구한다.
7. 최대 2000번, certificate 없이 끝나면 실패한다. feasible sum error≤1e−8, bounds tolerance1e−10을 추가 확인한다.

relaxed z를 그대로 threshold/top-m으로 반환하지 않는다. certified z와 Σ를 다음 FTRL normalization에 사용한다. actual run의 모든 certificate는 통과했다. relaxation tolerance, capped constraints, optimizer adaptation을 논문 Methods에서 숨기지 않는다.

### 15.3 Published FTRL sparsification

S=Σ(z*)라고 하고 symmetric inverse square root를 eigendecomposition으로 만든다. 다음을 정의한다.

\[
P_i=S^{-1/2}U_i,\quad D=S^{-1/2}(C_0/m)S^{-1/2},\quad
\widetilde H_i=D+P_iP_i^T.
\]

주의: 여기서 P_i는 vector이고 E-SDP의 dual P와 무관하다. η를 하나 고정하고 M=0으로 시작한다. slot k=0,…,m−1마다:

- k=0이면 `A_k^{-1/2}=sqrt(d)*I`.
- k>0이면 ηM=Q diag(e)Q.T. `shift=e-e_min`을 만들고 `sum_j (delta+shift_j)^(-2)=1`을 δ∈[1,sqrt(d)]에서 brentq xtol1e−12로 푼다. d=1은 δ=1. `nu=delta-e_min`, `A_k^{-1/2}=Q diag(delta+shift) Q.T`이다. 이는 `A_k=(nu I+eta M)^(-2), trace(A_k)=1`의 normalization이다.
- `Bhalf=(A_k^{-1/2}+eta D)^(-1)`.
- 유효 후보의 Eq.23 binary rank-one score는

\[
s_i=\frac{\|Bhalf\,P_i\|_2^2}{1+\eta P_i^T Bhalf\,P_i}.
\]

- s_i 최대 후보 j를 고르고 mask한다.
- M←M+D+P_jP_j.T, 대칭화한다.

정확히 m개의 distinct ID를 반환한다. target objective의 단순 greedy 감소량과 다르고 BAIT의 forward–backward pass도 수행하지 않는다. direct dense Eq.19 score와 rank-one Eq.23 score가 같은지 toy test로 확인했다.

### 15.4 η 선택: 확인된 공식 절차와 고정 grid

공식 supplement Appendix G.2는 선택된 normalized information sum의 minimum eigenvalue를 기준으로 learning rate를 선택한다. 배포 코드 run.py의 default eta는 200이다. 이번 실행 전 grid `[2.,20.,200.,2000.,20000.]`를 고정했다. grid 자체는 현재 저차원 benchmark를 위한 명시적 선택이며, paper에 동일한 다섯 값이 쓰였다는 주장은 하지 않는다.

각 eta마다 위 FTRL을 실행하고 다음 두 값을 저장한다.

- `normalized_min_eigenvalue = lambda_min(M_final)`.
- `internal_FIR_objective = trace((C0+sum_selected H_i)^(-1) F)`.

첫 번째 값을 최대화하는 eta를 선택한다. exact tie는 grid의 앞 값을 사용한다. metadata는 `eta_selection='published_G2_max_normalized_min_eigenvalue'`이다. 공식 절차를 찾지 못했을 때의 fallback 이름 `FIR_objective_only`를 붙이지 않았다. 두 번째 값은 audit용이며 eta 선택 criterion으로 바꾸지 않는다.

전 과정은 θ_ref, hidden label, endpoint 성능을 보지 않는다. eta별 subset, objectives, selected eta, relaxation z/S/gap/iteration, C0/F, FTRL slot score/nu, runtime을 batch audit에 남긴다.

### 15.5 수치 안정성과 adaptation 요약

rho I 때문에 C0와 Σ는 SPD이다. stable Cholesky/eigendecomposition을 사용한다. 새 run에서는 추가 jitter가 필요하지 않았고 모든 `jitter_events=[]`이다. 필요하다면 사전에 명시한 minimal jitter 정책과 실제 변화량을 기록해야 하며, 무단 ridge 증가로 solve failure를 숨기면 안 된다. 현재 코드는 non-SPD normalization을 감지하면 실패한다.

공식 실험의 multiclass logistic solver, class balancing, intercept convention, neural feature 또는 전체 pool target을 가져와 현재 estimator/query/candidate rule을 바꾸지 않았다. binary rank-one specialization, 기존 rho prior, capped weights, certified optimizer 교체, 현재 capped-pool target이 adaptation이다. FTRL matrix rule과 label-free G.2 selection은 유지했다.

## D. 실행과 저장 계약

각 (dataset,d,m,seed,method)를 독립 task로 실행한다. float64, BLAS thread 1, 24 process worker를 사용했다. 긴 Bernoulli/FIRAL 작업부터 queue에 넣어 idle time을 줄였다. 모든 scientific RNG는 process 수와 scheduling에 의존하지 않는다. 초기 데이터·reference는 read-only cache로 공유하며 trajectory의 관측 label prefix와 θ만 변한다.

각 completed batch마다 θ, signed error, 선택 prefix, information matrices/spectra와 JSON metadata를 atomic하게 저장한다. JSON commit과 array 파일이 모두 유효한 연속 checkpoint만 resume state로 인정한다. 완료 method의 기존 결과를 강제로 다시 실행하지 않는다. 결과가 누락/손상되면 해당 task를 따로 보고한다.

반드시 저장할 audit은 candidate IDs, selected IDs, frozen θ를 재구성할 수 있는 pre-batch checkpoint, RNG state, method/version, round, hyperparameters, runtime, numerical events이다. B-EMCM에는 bootstrap seed/model, RegVol에는 reverse removals, BADGE에는 D² draws 및 embedding norms, FIRAL에는 relaxation과 eta별 진단을 추가한다. endpoint는 evaluation namespace에 두고 selection function이 접근하지 못하게 한다.

## E. 최소 검증 세트와 실제 수행 결과

현재 `source/tests`에는 124개 통과 test가 있다. 새 `test_family_selectors.py`는 다음의 수학적 경계를 검증한다.

- B-EMCM: correlated toy에서 naive top-m과 다른 순차 correction, alpha=0 한계, bootstrap-scenario 독립 replay.
- LCMD: cluster mass=sum squared distance와 largest-cluster 선택; 단순 farthest-first와 구별.
- RegVol: tiny matrix의 reverse Markov distribution 열거와 7,000 draw의 subset frequency, exact cardinality/replay/label independence.
- Entropy: 극단 logits에서도 finite, p=0.5가 최고.
- FIRAL relaxation: 독립 constrained SLSQP objective와 relative 2e−7 이내 일치, bounds/sum/gap certificate.
- FIRAL FTRL: direct dense matrix criterion과 binary rank-one score의 slot별 일치, exact cardinality.
- FIRAL eta: normalized minimum eigenvalue의 argmax rule 일치, no-reference dependence.
- 실제 cutoff/candidate 정책과 nonuniform full-pool target을 가진 toy에서 BAIT와 FIRAL이 다른 subset을 선택함.
- 모든 새 method: 동일 RNG state의 deterministic replay, family eligibility, forbidden evaluation input에 의존하지 않는 interface, label-free method의 model/label independence.

실제 데이터 smoke는 5 methods × 3 m × 3 d, seed0의 첫 2 batch를 수행했다. 이후 전체 450 task를 실행하고 초기 checkpoint가 이전 ours와 bytewise 같은지, candidate 재생성 및 selected prefix가 일치하는지, 모든 fit/metric이 완료됐는지 확인했다. BADGE는 existing implementation tests와 저장된 완성 trajectory를 재검사했다. 추가 method의 전체 ordered trajectory가 기존 comparator와 동일한 경우는 0건이었다.

## F. Implementation location map

| 표시 이름 | 내부 method ID | 실제 함수 / source |
|---|---|---|
| ours | fb_opt2 | selectors.py: fb_opt2_scale, fb_opt2_temperature, select_batch |
| A-opt | a_opt | selectors.py: select_batch + InverseCache |
| D-opt | d_opt | selectors.py: select_batch + InverseCache |
| E-greedy | e_greedy | selectors.py: _e_greedy, e_scores_batched, e_scores_secular |
| E-SDP | e_sdp | selectors.py: _e_sdp, project_capped_simplex, dependent_round, dual_upper_bound |
| BAIT | bait_glm | selectors.py: select_batch bait_glm branch + InverseCache |
| UCB | ucb_glm | selectors.py: select_batch ucb_glm branch |
| TS | ts_glm | selectors.py: select_batch ts_glm branch |
| Random | random | selectors.py: select_batch random branch |
| B-EMCM | b_emcm | family_selectors.py: bemcm_select, select_family_batch bootstrap branch |
| LCMD | lcmd | family_selectors.py: lcmd_select |
| RegVol | regvol | family_selectors.py: regvol_select |
| Entropy | entropy | family_selectors.py: select_family_batch entropy branch |
| BADGE | badge | selectors.py: select_batch badge branch (재사용) |
| FIRAL | firal | family_selectors.py: fir_relax, firal_ftrl, select_family_batch firal branch |

모든 source 경로는 패키지의 `source/real_response/` 기준이다. common.py의 seed/hash 규약, runner.py의 batched feedback, numerics.py의 fitting, evaluation.py와 reporting.py의 metric을 함께 보아야 한다. 새 방법을 잘 구현해도 objective의 sum/mean scaling, intercept penalty, seed pairing, Q의 uncentered convention을 바꾸면 이 결과를 재현하지 못한다.
