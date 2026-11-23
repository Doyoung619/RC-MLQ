# Codex 실행 프롬프트: 완료된 RC-MLQ × Meta-World 실험에 M=2 benchmark만 추가

## 0. 이번 작업과 우선순위

현재 repository의 완료된 **10-interaction Meta-World 실험**을 기준으로, **M=2인 동일 setting에만** 아래 benchmark를 추가 구현·검증·실행하고, 기존 결과와 병합한 그림·표·보고서를 생성하라.

이것은 전체 실험 재실행 요청이 아니다. **M>2의 statistical runs와 rollout을 다시 실행하지 마라.** 기존 결과, 실행 코드, RNG mapping, q/y history, simulator context를 보존하라. 이 프롬프트는 첨부된 이전 `RC-MLQ × Meta-World 고속 실험` 명세의 **추가 실행 명세**이며, 충돌하는 부분만 아래처럼 대체한다.

1. 사용자의 최신 지시가 최우선: **10 interactions**, **M=2만 추가**.
2. 완료된 run의 frozen manifest가 실제 seed, theta_star, context, adapter, package version의 기준이다. 이전 프롬프트의 `5 seeds`, `context_count: 5`, 단일 GPU workstation 가정은 현재 실행 명세로 복사하지 마라.
3. 아래에 명시한 새 benchmark 정의 및 source provenance를 사용한다. 기존 `SG` 함수를 덮어써서 다른 알고리즘으로 만들지 마라.
4. 그 외 likelihood, inference target, T, d, robot realization, numerical tolerance, GPU 제한은 완료된 실험과 동일하게 유지한다.

**최종 목표는 실제로 누락된 M=2 jobs만 실행하여 끝내는 것이다.** 문서·TODO·스켈레톤만 만들고 끝내지 마라. 실행 환경이나 권한이 없는 경우에는 실행했다고 주장하지 말고, 구현·검증한 부분과 정확한 blocker를 구분하라.

### 0.1 범위를 고정한다

추가 대상 setting은 원래 일곱 setting 중 하나인 다음 setting뿐이다.

```yaml
setting_id: m02_b04
dimension: 4
choices: 2
beta_response: 4.0
beta_inference: 4.0
answers_per_run: 128
interaction_count: 10
prior: Uniform([-1, 1]^4)
```

새 M=2 beta sweep, 새 dimension sweep, T=4096 실험, RL 학습을 자동으로 추가하지 마라. 실제 완료 manifest가 이 설정과 다르면 덮어쓰기 전에 차이를 보고하고, 같은 setting으로 식별되지 않는 결과를 합치지 마라.

### 0.2 이 프롬프트에서 말하는 “원 논문 그대로”의 정확한 범위

새 Gaussian baseline은 **Canal et al.의 공개 코드가 사용하는 finite-stage budget allocation, stage별 표준편차 감소, stage 경계에서의 재중심화 구조**를 따른다. 직접 확인한 원본은 마지막의 [G-code]다.

그러나 이번 실험은 원래의 catalog가 없고, 모든 방법에 공통 Bayesian inference와 fixed-beta squared-distance logit을 유지한다. 따라서 **원 논문의 전체 추정기·catalog 절차까지 복제한 실험이라고 쓰면 안 된다.** 정확한 이름은 다음과 같다.

> GaussCloud-Q (author-code schedule; continuous/shared-Bayes transfer)

원본 코드에서는 catalog의 분산으로 초기 scale을 정한다. catalog가 없는 이번 실험에서 이를 어떻게 옮기는지는 Section 4.3에 **명시적인 새 설정 선택**으로 고정했다. “원 논문이 Meta-World에서 이 수치를 사용했다”라고 쓰지 마라.

근거를 `[BASE]`(완료된 실험/제공 원고), `[SOURCE]`(원 논문/저자 코드), `[TRANSFER]`(현재 모델로의 이식), `[ENGINEERING]`(구현 및 평가의 새 선택)로 구분하여 `PROVENANCE.md`에 기록하라.

---

## 1. 실행 전: 완료된 실험을 잠근 뒤 재사용 계획부터 만든다

### 1.1 Repository와 artifact 조사

`AGENTS.md`, git status, 현재 실행 중인 관련 작업, 기존 Meta-World runner/selector/inference/plotting 코드를 읽어라. 기존 작업을 reset, clean, delete하지 마라.

다음 자료를 실제 경로로 찾아 `BASELINE_LOCK.json`에 기록한다.

- 완료된 10-interaction experiment manifest와 frozen config.
- `m02_b04`의 interaction IDs와 실제 RNG seeds, theta_star mapping.
- 기존 RC/PQ/QI/PS/GC/SG의 M=2 완료 여부와 q/y history.
- posterior checkpoint, initial representation, inference backend/tolerances.
- Meta-World environment/task/version, nominal action sequence, context restore state.
- H=50, 네 motion basis, global action gain, gripper 값, adapter code/hash.
- existing numerical validation과 executed rollout/cache records.
- 각 metric의 실제 denominator와 저장 field 의미.

파일을 찾지 못했다고 zero-filled 결과를 만들거나, figure 이미지를 수치 artifact로 사용하지 마라.

### 1.2 기존 seed와 latent를 보존한다

기존 10개 interaction의 ID를 그대로 사용한다. 기존 ID가 `[0,...,9]`이면 이를 사용하고, 다른 ID면 실제 ID를 보존한다. 새로운 root seed로 전체 latent를 다시 뽑지 마라.

각 새 method는 같은 `(setting, interaction_id)`의 기존 theta_star를 **값까지 동일하게** 공유한다. selector/response/inference/diagnostic RNG는 method와 purpose별로 분리한다. 등록 순서나 worker 배치 때문에 기존 method의 seed가 달라져서는 안 된다.

기존 latent가 seed마다 다른지, 여러 seed에서 고정 theta_star를 쓰는지 manifest로 확인하고 그대로 유지한다. 전자는 prior-averaged 해석, 후자는 그 고정 preference에 조건부인 실험이다. 10 seeds라는 이유만으로 10개의 독립 latent draw가 있었다고 쓰지 마라.

기존 10-seed 실험의 context mapping도 보존한다. 반드시 10개의 서로 다른 simulator context였다고 추측하거나, 기존 5개 context를 몰래 다시 매핑하지 마라. 추가 method는 대응 interaction에서 기존과 같은 context를 사용한다.

### 1.3 기존 결과의 불변성을 검사한다

최소한 완료된 manifest, frozen config, q/y history, 주요 statistical arrays, context/adapter files의 SHA-256 inventory를 작업 전후 비교한다. 공유 코드 수정은 작고 backward-compatible하게 한다. 새 결과는 별도 output root에 저장한다.

기존 SG는 `SG-legacy`라는 **reporting alias**로 보존하고 기존 method ID는 변경하지 마라. 새 GaussCloud-Q는 별도의 IDs로 추가한다. 기존 SG가 잘 나왔다는 이유로 그 기록을 삭제하거나, 새 알고리즘 이름으로 재라벨링하지 마라.

---

## 2. 최종 method registry: 14개 configuration

### 2.1 기존 M=2 control 5개

아래 다섯 방법은 완료된 compatible M=2 결과를 우선 재사용한다. 완료되지 않은 interaction만 기존 정의로 보충한다.

| ID 예시 | 표기 | 현재 실험에서의 정의 |
|---|---|---|
| 기존 RC ID | RC | 기존 median-centered Fisher RC-MLQ, 7-scale grid |
| 기존 PQ ID | PQ | widest projection의 midpoint posterior quantiles |
| 기존 QI ID | QI-1 | 자기 posterior의 Fisher-RC seed에서 한 번의 hypothetical-mean transform |
| 기존 PS ID | PS | 자기 joint posterior에서 independent draws, with replacement |
| 기존 GC ID | GC | mean-centered Gaussian with full covariance Sigma_t, fixed cap 32 |

단순한 Gaussian covariance baseline인 GC를 published GaussCloud-Q의 원본이라고 설명하지 마라. RC/PQ/QI/PS/GC를 이번 추가 실험에 맞춘다는 이유로 변경하지 마라.

### 2.2 제공 원고의 binary-only 비교방법 6개를 전부 추가한다

| 새 ID 예시 | 표기 | 범위/출처 |
|---|---|---|
| `sm_continuous` | SM | SAMPLEMIRROR proposal geometry, shared-posterior continuous transfer [SM] |
| `mem_line` | MEM-line | binary MEM criterion, 원고의 coordinate-line specialization [MEM] |
| `gmem_line` | GMEM-line | 같은 line에서 첫 후보를 posterior mean으로 고정한 greedy MEM [MEM] |
| `2mem_line_2sweeps` | 2MEM-line (2 sweeps) | GMEM 초기화, slots 1,2,1,2 갱신 [BASE, MEM] |
| `infosynth_trace` | IT | Info-Synth trace-scale component transfer [IS] |
| `infosynth_bounded` | IB | 원고의 bounded-relative Info-Synth transfer [BASE, IS] |

이는 제공 원고의 binary 비교 목록을 전부 추가하는 것이다. 다른 인용 논문의 전체 benchmark 목록까지 임의로 가져오는 작업이 아니다. EPMV, MCMV, ActRank, ContFree 계열을 자동으로 추가하지 마라. 실제 repository의 최신 원고 목록이 다르면 차이를 기록하고, 명시된 여섯 방법은 빠뜨리지 마라.

**MEM의 line implementation을 full-dimensional global MEM이라고 부르지 마라.** SM/IT/IB를 원 논문의 end-to-end exact reproduction이라고 부르지 마라.

### 2.3 새 finite-stage Gaussian 비교방법 3개

Canal et al. Figure 2가 비교하는 stage counts 1,3,5를 그대로 사용하여 세 configuration을 모두 실행한다. 여기서 stage count를 L로 표기한다. `M=2`의 후보 수와 구분하라.

```text
GCQ-1: L=1
GCQ-3: L=3
GCQ-5: L=5
```

method IDs는 예컨대 `gausscloud_paper_L1`, `gausscloud_paper_L3`, `gausscloud_paper_L5`로 둔다. 이 세 개는 동일한 방법의 stage-count configurations이며 세 개의 독립적인 학술 알고리즘으로 세지 마라. 결과를 본 뒤 유리하거나 불리한 하나만 선택하지 마라.

기존 SG-legacy는 추가 실행하지 않고, 이미 존재하면 별도 schedule-comparison appendix에서 재사용한다. 새 14-configuration 기본 registry에는 포함하지 않는다.

### 2.4 계산량과 재사용

필요한 최종 기본 registry는 `(5 + 6 + 3) × 10 = 140`개의 M=2 run이다.

기존 control 다섯 방법이 각 10개씩 완료되어 있고, 새 아홉 configuration의 compatible 결과가 없으면:

```text
새 statistical runs           = 9 × 10 = 90
새 실제 preference answers   = 90 × 128 = 11,520
새 selected candidate clips  = 11,520 × 2 = 23,040
H=50 논리적 env.action steps = 23,040 × 50 = 1,152,000
```

검증·diagnostic·video 비용은 별도다. 이것은 실제 runtime 추정이 아니다. 기존 binary controls나 새 방법의 compatible run이 이미 있으면 재사용하여 신규 계산량을 줄인다. 최악의 기본 registry를 전부 새로 계산해야 하면 140 runs, 17,920 answers, 35,840 clips다. `REUSE_PLAN.json`에는 required/reused/new/resumed/missing를 실제로 계산해 적는다.

**T에 의존하는 GCQ-L은 T=4096 run의 첫 128개를 재사용할 수 없다.** T가 stage boundaries에 들어가므로 config hash에 T를 반드시 포함한다. control도 prefix가 동일한 selection law와 numerical history를 따랐을 때만 재사용한다.

---

## 3. 변경하지 않는 통계 모델과 인덱싱

Preference parameter는 theta_star, displayed candidate는 q_m, posterior mean/covariance는 mu_t/Sigma_t, largest posterior SD는 sigma_t, top unit direction은 v_t로 쓴다. 이 네 좌표는 robot joint vector가 아니라 기존 motion adapter의 입력이다.

각 질문의 두 후보에 대한 실제 likelihood는 다음이다.

\[
\ell_m(\theta,Q)=\frac{\exp[-\beta\|\theta-q_m\|^2]}{\sum_{j=1}^2\exp[-\beta\|\theta-q_j\|^2]},\qquad \beta=4.
\]

동치 logits `2*beta*(theta @ q_m) - beta*||q_m||^2`를 log-domain으로 계산한다. 후보별 norm term을 빼거나 pair difference를 길이로 정규화하지 마라. 모든 실제 response와 Bayesian update에 **동일한 displayed Q**를 사용한다.

각 run은 같은 prior에서 시작하여 자기 actual answers 128개만 관측한다. 다른 method의 posterior를 빌리거나, 기존 RC의 최종 posterior에서 새 baseline을 시작하지 마라. 진짜 theta_star는 oracle/error evaluator만 접근한다.

- pre-answer t=0,...,127: selector, query EIG, radius, diagnostic metadata.
- post-answer states t=0,...,128: full 4-coordinate covariance trace와 actual squared parameter error.
- `posterior_risk = tr(Sigma_t)`와 `realized_sq_error = ||mu_t-theta_star||^2`는 별도 field다.
- t=128의 dummy query를 만들지 마라.
- 초기 analytic prior는 mu_0=0, Sigma_0=I_4/3, trace=4/3이다.

새 방법의 inner optimization과 diagnostic query는 **추가 사용자 답변이 아니다.** actual history를 갱신하지 않는다.

---

## 4. GaussCloud-Q: 저자 공개 코드의 schedule을 정확히 구현

### 4.1 확인한 원본과 다른 서술을 섞지 마라

선택한 구현 원본은 [G-code]의 `enusvm/nusvm_sim.m`이다.

```text
repository: siplab-gt/pairsearch
commit: 9e621462af6b7ca502b362b1415c781dc10eb404
file: enusvm/nusvm_sim.m
Git blob SHA: 1969bfe73ca95027060b22ab0d2421cc446d7941
```

실행 전에 이 pinned text를 확인하고 local source snapshot 또는 hash/line anchors를 저장한다. 원본 경로·허용 라이선스를 확인하지 않고 repository 전체를 제품 코드에 복사하지 마라. 논문/코드가 일부 세부사항에서 다르면 둘을 별도로 기술한다.

**직접 확인된 코드의 규칙:**

- 각 stage에 `floor(T/L)`개를 먼저 할당한다.
- 남는 comparisons를 **마지막 stage에 전부 더한다**.
- stage index j=0,...,L-1에 대해 **표준편차**를 `s0 * 2**(-j)`로 둔다. covariance는 `s0**2 * 4**(-j) * I`이다.
- stage 시작의 추정치로 cloud를 재중심화하고 그 stage의 proposals는 그 중심으로 생성한다.
- 원본은 `s0 = max coordinate sample SD of catalog Y`를 사용한다.
- 원본은 catalog retrieval/중복 pair 처리 및 enusvm estimator를 사용한다. 이번 continuous/shared-Bayes transfer는 이 두 부분을 동일하게 재현하지 않는다.

Massimino–Davenport Section 5.2의 이론적 radius-halving, Section 6.4의 variance 표현, 다른 실험의 rounding 관행을 한 구현 안에서 임의로 섞지 마라. **이번 default의 std 감소·rounding 기준은 위 pinned author code**다.

### 4.2 T=128에서의 고정 stage allocation

\[
n_j=\lfloor T/L\rfloor\ (j<L-1),\qquad
n_{L-1}=T-(L-1)\lfloor T/L\rfloor.
\]

따라서 반드시 다음 표와 일치해야 한다.

| L | stage lengths | 각 stage의 pre-answer t 시작점 |
|---|---|---|
| 1 | [128] | [0] |
| 3 | [42,42,44] | [0,42,84] |
| 5 | [25,25,25,25,28] | [0,25,50,75,100] |

`np.array_split`의 앞쪽 remainder 배분으로 대체하지 마라. L=3을 `[43,43,42]`로 만들지 마라. L=5를 `[26,26,26,25,25]`로 만들지 마라.

간결한 scheduler 동작 예시는 다음과 같다. 실제 구현은 기존 상태/체크포인트 interface에 맞춰라.

```python
base = T // L
lengths = [base] * L
lengths[-1] += T - sum(lengths)
starts = np.cumsum([0] + lengths[:-1])
j = np.searchsorted(starts, t, side="right") - 1
if j != state.stage:
    assert t == starts[j]  # resume는 저장 state를 복원한 뒤 호출
    state.center = posterior.mean().copy()
    state.stage = int(j)
std = s0 * (2.0 ** (-j))
raw_pair = state.center[None, :] + std * selector_rng.normal(size=(2, d))
```

매 round posterior를 update하되, stage 내부에서는 proposal center와 std를 그대로 유지한다. 다음 stage는 직전 stage의 모든 actual answers를 반영한 posterior mean에서 시작한다. diagnostic 호출로 stage transition을 일으키지 마라.

### 4.3 초기 scale: source rule과 이번 이식을 분리하여 고정

원본 코드의 catalog Y가 이번 연속 q 실험에는 없다. 외부 Food-10k embedding을 몰래 가져와 q 공간으로 사용하거나, 현재 posterior covariance를 source catalog covariance라고 부르지 마라.

이번 specification의 명시적 `[TRANSFER]` 기본값은 **고정된 초기 reference distribution p0의 최대 marginal SD**를 source의 catalog SD에 대응시키는 것이다.

\[
s_0=\max_i\sqrt{\operatorname{Var}_{p_0}(\Theta_i)}=1/\sqrt3.
\]

이는 `Uniform([-1,1]^4)`에서 analytic하게 계산되며, seed별 sample SD나 현재 posterior width로 다시 추정하지 않는다. 세 L configuration에 동일한 s0를 사용한다. 이전 SG-legacy의 s0=1을 이 새 source-scale mapping과 혼동하지 마라. **Legacy SG와 새 GCQ의 차이는 schedule뿐 아니라 초기 scale mapping도 포함한다.** 그 성능 차이를 schedule 하나의 인과효과라고 설명하지 않는다.

`PROVENANCE.md`에 다음을 명시한다.

```text
initial_scale_source_rule: max_coordinate_sd_of_fixed_catalog
initial_scale_transfer: analytic_max_coordinate_sd_of_initial_uniform_prior
initial_std: 1/sqrt(3)
initial_scale_is_an_adaptation: true
```

이 수치 선택은 원 논문 실험의 수치를 복제한 것이 아니라 **catalog-free task로 옮기는 결정**이다. 결과를 보고 s0, L, shrink factor를 바꾸지 마라. exact catalog experiment를 요구하는 별도 재현 작업과 이번 shared-setting 실험을 혼합하지 마라.

### 4.4 Shared-Bayes 이식과 cap

Raw pair는 stage center 주변의 independent d-dimensional isotropic Gaussian pair이다. EIG reranking, principal-axis 제한, per-round recentering을 넣지 마라.

Actual displayed pair에는 기존 GC/SG와 동일한 current-posterior-mean 중심 radius-32 projection을 적용한다. 원본에는 이 동일한 cap이 없으므로 `[TRANSFER]`로 기록하고 raw/displayed pair와 cap activation을 모두 저장한다. cap을 posterior-width 비례로 바꾸지 마라.

원본의 enusvm estimator는 이번 기본 comparison에서 사용하지 않는다. 현재 posterior mean으로 stage center를 구하는 것은 명시적인 shared-estimator transfer이다. simulator 후보 catalog, nearest-neighbor retrieval, 원본 noisy likelihood의 normalized/decaying 옵션을 추가하지 않는다. 원본의 3배 oversampling과 catalog deduplication도 continuous pair sampler에 기계적으로 붙이지 마라.

### 4.5 기존 SG와의 비교 및 이론 범위

기존 SG-legacy는 `floor(log2(1+t/128))`를 사용하므로 T=128의 actual queries에서는 **stage 0, std=1, 최초 center**를 유지한다. 현재 짧은 실험에서 좋은 성능을 보였다는 사실만으로 정교한 stage tuning 효과라고 해석하면 안 된다. 새 GCQ-3/5는 실제로 T=128 안에서 scale과 center가 바뀐다.

고정 L finite-horizon GCQ-L을 자동으로 `U_C` 또는 `U_{C,epsilon}`의 asymptotic theorem member라고 표시하지 마라. 마지막 positive scale로 계속하는 연장은 `s_t -> 0`가 아니다. T를 바꿀 때 boundaries도 달라지는 horizon-indexed family를 하나의 고정 infinite policy처럼 취급하지 않는다.

기존 SG-legacy를 무한히 계속한 mathematical selector는 vanishing scheduled spread로 broad class에 들어가는 예시지만, 그것과 새 GCQ-L은 별개의 policy이다. 이 추가 실험에서 어느 GCQ curve도 반드시 information ratio가 0으로 가야 한다거나 MSE가 T^-1/2여야 한다고 강제하지 마라.

---

## 5. Binary-only methods의 구현 명세

### 5.1 SM: continuous SAMPLEMIRROR geometry

현재 posterior mean/covariance를 mu/Sigma, top unit eigenvector를 v라 하자. 한 joint Gaussian vector z를 그 covariance로 샘플링한다.

\[
z\sim\mathcal N(0,\Sigma),\qquad
q_1^{raw}=\mu+z,\qquad
q_2^{raw}=\mu+z-2(v^\top z)v.
\]

두 개의 independent Gaussian을 뽑는 것이 아니다. `mu-z`를 두 번째 후보로 쓰는 point reflection도 아니다. principal hyperplane에 대한 반사이며, v에 수직인 공통 성분을 유지한다. [SM, Algorithm 1]

기존 current-mean radius-32 cap을 동일하게 적용한다. 양쪽 raw centered norm이 같으므로 동일 radial factor가 적용되는지 검사한다. Oracle/EIG/update는 displayed pair로 한다.

원본의 catalog retrieval, unused-item 제한, normalized probit likelihood, assumed-density Gaussian posterior를 가져오지 않는다. 이 차이를 표기한다. M!=2를 명시적으로 거부한다.

Diagonal product posterior와 coordinate v에서는 두 후보가 한 coordinate에서만 달라지고 나머지는 동일하다. cap 이후에도 실제 displayed Q로 이 조건을 검사하면 기존 marginal inference를 사용할 수 있다. full-dimensional raw z를 뽑았다는 이유만으로 반드시 joint inference가 필요한 것도 아니고, 반대로 SM이라는 이름만으로 무조건 factorized라고 가정할 수도 없다.

### 5.2 MEM / GMEM / 2MEM: 현재 원고와 같은 binary coordinate-line specialization

이번 작업은 기존 원고의 coordinate-line experiment를 구현한다. candidate domain을 현재 posterior width에 비례하는 작은 영역으로 줄이지 마라.

각 round에 product posterior의 최대 분산 coordinate k를 고른다. 동률은 기존 규칙을 따른다. 그 coordinate의 가능한 값은 원래 domain 전체 `[-1,1]`이고, 나머지 candidate coordinates는 현재 mean에 고정한다.

선택된 coordinate의 posterior variable을 X, 그 marginal의 positive normalized integration nodes/weights를 `(x_i,w_i)`라고 하자. scalar centers a,b의 objective는

\[
D_t(a,b)=\mathbb E[\min\{(X-a)^2,(X-b)^2\}]
       \approx\sum_i w_i\min\{(x_i-a)^2,(x_i-b)^2\}.
\]

이는 expected-best-candidate quadratic loss이지 EIG가 아니다. 실제 답변은 여전히 공통 noisy logit이다. 두 scalar centers는 `q_1=mu+(a-mu_k)e_k`, `q_2=mu+(b-mu_k)e_k`로 d차원 후보에 넣는다.

**MEM-line:** a,b in [-1,1]를 함께 최적화한다. 1D sorted weighted partition enumeration을 사용하면 frozen discrete objective의 global solution을 구할 수 있다. 모든 유효한 split에 대해 양쪽 weighted mean과 within-cluster squared loss를 prefix sums로 계산하고 가장 작은 loss를 택한다. boundary/zero-mass/degenerate cases와 deterministic ties를 처리한다. continuous posterior의 exact population optimizer라고 과장하지 마라. quadrature refinement로 numerical error를 확인한다.

**GMEM-line:** 첫 center a=E[X]=mu_k를 고정하고 b in [-1,1]를 전역 최적화한다. b의 탐색 범위를 posterior quantile interval로 줄이지 마라.

**2MEM-line:** 정확하게 풀린 frozen discrete GMEM solution으로 초기화하고, 동일 posterior에서 slot order **(1,2,1,2)**로 네 번의 one-center update를 한다. 즉 원고의 **두 sweeps**다. 각 update에서 다른 center를 고정하고 전체 [-1,1]를 검색한다. objective는 증가하면 안 된다. 원 [MEM]의 until-no-progress version과 달리 이번 two-sweep budget은 원고의 지정 버전이라고 표시한다. 추가 convergence run을 기본 registry에 몰래 넣지 마라.

#### One-moving-center global solver의 안전한 구현

fixed center f와 moving center b의 scalar objective는 각 node에서 두 제곱거리의 minimum이다. b 영역의 breakpoints는 `b=f`, `b=2*x_i-f`, 그리고 domain endpoints이다.

1. [-1,1] 안의 breakpoints를 정렬하여 닫힌 intervals를 만든다.
2. interval interior의 대표점에서 어느 nodes가 moving center에 할당되는지 결정한다. b>f이면 오른쪽 suffix, b<f이면 왼쪽 prefix다.
3. 그 interval에서 moving center의 최적값은 해당 nodes의 weighted mean을 interval에 투영한 값이다. 해당 mass가 0이면 interval endpoint 등의 deterministic 대표를 사용한다.
4. interval minimizers, domain endpoints, b=f를 실제 min-distance objective로 평가하여 최솟값을 반환한다.
5. Prefix moments를 사용하여 반복 node loops를 줄인다. 무검증 local scalar optimizer 한 번으로 전역해를 주장하지 마라.

이것은 [MEM]의 일반 알고리즘이 원래 이렇게 구현되어 있다는 주장이 아니라, 이번 frozen 1D numerical objective를 푸는 **derived numerical solver**다.

### 5.3 IT: Info-Synth trace-scale transfer

현재 posterior의 전체 covariance trace의 제곱근을 `s = sqrt(tr(Sigma))`라고 하자.

\[
q_1=\mu-sv,\qquad q_2=\mu+sv.
\]

모든 4 coordinates의 trace를 사용한다. sigma_t 또는 선택 coordinate SD로 대체하지 마라. scalar refinement를 추가하지 않는다. “full Info-Synth”가 아니라 **trace-scale component transfer**다. [IS, Appendix B.3; BASE]

이 설정에서는 trace <=4이므로 s<=2이고 cap 32 이내다. unnecessary clipping을 넣지 마라.

### 5.4 IB: 원고의 bounded-relative Info-Synth transfer

IT와 같은 `s=sqrt(tr(Sigma))`를 사용하고, relative half-spacing을 x=r/s라고 정의한다. 보조 Gaussian variable은 `U ~ N(0, Sigma/tr(Sigma))`이며, `sigma0=0.25`는 **acquisition 내부의 confidence parameter**다. 실제 beta=4 response와 혼동하지 마라.

logistic function을 `logistic(z)=1/(1+exp(-z))`, binary entropy를 `h(p)=-p log p-(1-p) log(1-p)`라 두고 다음 목적함수를 최소화한다.

\[
F_t(x)=\mathbb E_U\left[h\left(\operatorname{logistic}
\left(\frac{4x v^\top U}{\sigma_0\sqrt{2(\|U\|^2+x^2)^2+8x^2(v^\top U)^2}}\right)\right)\right].
\]

최종 후보는 `mu ± (s*x_selected)*v`이다. 이것은 [BASE]가 지정한 conditional-entropy acquisition이며 현재 actual posterior의 exact EIG maximization이 아니다.

설정은 원고에 맞춰 다음으로 고정한다.

```yaml
sigma0: 0.25
relative_min: 0.01
relative_max: 10.0
physical_halfspacing_cap: 32.0
log_grid_points: 25
sobol_points: 1024
reference_sobol_points: 4096
refine_discrepancy_threshold: 0.0005
log_scale_tolerance: 0.00001
```

매 trial, refinement, boundary evaluation에서 `x in [0.01, min(10,32/s)]`를 지킨다. Grid의 모든 local minimum 주변을 bounded refinement하고 두 domain endpoints도 평가한다. fixed common scrambled Sobol Gaussian draws를 한 acquisition call 안에서 공유하고, objective evaluation마다 random points를 다시 뽑지 마라. Gaussian symmetry에 따른 eigenbasis integration을 이용할 수 있다.

entropy는 overflow에 안정적인 수식으로 계산한다. 최소화할 h와 음의 entropy를 혼동하지 마라. source의 서로 다른 표기에서 sign이 불명확하면 현재 원고의 **conditional entropy 최소화**를 기준으로 audit note를 남긴다.

bound 10, 보조 Gaussian belief, sigma0, quadrature 및 refinement는 명시된 adaptation이다. unconstrained original Info-Synth를 재현했다고 쓰지 마라. fixed particle search가 `U_C`의 exact-posterior 가정을 자동으로 증명하는 것도 아니다.

---

## 6. 기존 inference를 재사용하되 실제 query geometry로 분기한다

### 6.1 Marginal engine 후보

기존 RC/PQ/QI, 새 MEM/GMEM/2MEM/IT/IB 및 조건을 만족하는 SM은 같은 positive-density adaptive marginal implementation을 우선 사용한다.

- initial 257 nodes, 검증에 513 nodes 또는 기존의 더 정확한 resolution.
- [-1,1] 지원, mesh Jacobian, log-domain update, continuous CDF inversion 유지.
- remesh는 그 coordinate의 **전체 actual likelihood history**를 다시 반영.
- numerical nonpositive/negative weights를 probability weights로 쓰지 않음.

이는 한 질문에서 두 후보가 한 coordinate에서만 다를 때만 허용된다. 공통 tangent coordinate가 있어도 같으면 likelihood에서 취소된다. 실제 displayed pair로 factorization을 검사하라.

### 6.2 Joint engine

PS/GC/새 GCQ-L은 기존 joint tempered resample-move SMC를 사용한다. 기존 validated posterior engine을 새로 단순화하지 마라.

```text
final particles = 4096
reference particles = 8192 이상 필요시 단계적 refinement
tempering target ESS >=0.8N
Metropolis sweeps per resampling =10
final resampling only when ESS<0.5N
posterior target = full actual history under shared beta=4 model
```

bounded prior 밖 particle proposals를 reject한다. likelihood를 일부만 반영하고 종료하거나 초기 particles를 reweight만 반복하는 구현은 금지한다. Gaussian delayed-acceptance screen을 써도 exact full-history correction을 유지한다.

### 6.3 검증 및 이론 분리

이론은 지정된 exact-posterior policy에 대한 것이고 실험은 numerical approximation이다. `exact_eig_formula_used`와 `exact_posterior_computed`를 구분하라.

대표 interaction을 ID 순서로 결과를 보기 전에 고정하고, frozen actual histories의 t=0,32,64,128을 doubled-resolution/reference engine으로 재평가한다. 기본 tolerance는 variance/EIG relative 0.005, information-ratio absolute 0.005이며 stochastic reference uncertainty도 기록한다. 새로운 adaptive queries를 생성해서 다른 history와 비교하지 마라.

---

## 7. 지표: 새 I* 정의를 롤백하지 않고 기존 수치와 연결

### 7.1 모든 round에서 기존 값은 eta_grid로 보존

제공된 이전 Meta-World prompt는 7개 line-grid EIG 최대값을 denominator로 사용했다. 이 값을 새 ellipsoidal oracle과 혼동하지 않는다.

모든 method 자신의 pre-answer posterior에서 기존 7-scale template을 다시 만들고, integrated EIG를 계산한다.

\[
I_t^{grid}=\max_{r\in\{0.125,0.2,0.32,0.5,0.8,1.25,2.0\}}I_t(Q_{t+1}(r)),
\qquad \eta_{t,grid}=I_t(Q_{t+1}^{display})/I_t^{grid}.
\]

main numerical comparison은 **기존 결과와 동일한 `eta_grid`와 full posterior covariance trace**로 유지한다. 이는 새 theorem의 I*를 이전 정의로 되돌리는 것이 아니다. 기존 dataset의 `eta` field를 원본에서 수정하지 말고, 새 reporting adapter에서 `eta_grid`로 매핑하며 denominator metadata를 기록한다.

Fisher score, predictive entropy, `0.5*sigma^2*J`를 integrated EIG로 대체하지 않는다. small KL의 기존 float64 안정화와 unresolved flag를 유지한다. ratio>1을 삭제하거나 clip하지 마라.

### 7.2 최신 full-dimensional oracle은 별도 평가 sidecar

수정된 원고의 oracle 정의를 읽고 **ellipsoid를 임의로 ball로 바꾸지 마라.** 완성된 run/config에 별도 oracle evaluation 설정이 이미 있으면 그 값을 계승한다.

없으면 이번 M=2-only evaluation의 명시적 default는 다음이다. 단위행렬은 I_4, prior domain의 diameter는 D=4, binary template bound는 B_2=1+sqrt(3), 최대 grid scale은 2다.

\[
\alpha_E=1/2,\qquad \rho_E=D/2+B_2\,2=4+2\sqrt3,
\]
\[
W_t=\alpha_E I_4+(1-\alpha_E)\Sigma_t/\sigma_t^2,\qquad
\mathcal E_t=\{\mu_t+\rho_E W_t^{1/2}z:\|z\|\le1\}.
\]

이는 M=2에서 current RC grid를 포함하는 fixed-radius choice다. 모든 method/seed/round에 같은 alpha_E/rho_E를 사용한다. motion adapter의 gain alpha와 전혀 다른 parameter다. 기존 M>2의 oracle 값을 이 radius로 다시 계산하지 않는다.

새 이론의 reference는 두 후보가 이 ellipsoid 안에 있는 **모든 feasible pairs**의 최대 integrated information이다.

\[
I_t^{star,E}=\max_{q_1,q_2\in\mathcal E_t,\;(q_1,q_2)\in\mathcal Q}I_t(q_1,q_2).
\]

이번 domain에서는 rho_E<32라서 E_t 안의 모든 pair가 fixed-cap feasible domain에 들어간다. 그러나 actual baseline Q가 E_t 밖일 수도 있다. baseline을 E_t에 새로 project하지 마라.

### 7.3 전역값인 척하지 않는 M=2 oracle diagnostics

최소한 매 round analytic upper bound와 feasible-grid lower bound를 sidecar에 저장한다. frozen numerical posterior에 대한 candidate-query lower bound를 L_t, analytic upper bound를 U_t라 하자.

\[
L_t\ge I_t^{grid},\qquad
U_t=\min\{\log2,\ 2\beta^2\rho_E^2\operatorname{tr}(\Sigma_t)\}.
\]

upper bound는 ellipsoid inside rho_E-ball이라는 사실과 원고의 general information bound로부터 얻는다. lower bound에는 **같은 posterior에서 같은 ellipsoid 안임을 확인한** feasible query의 EIG만 사용한다.

**Full ellipsoid optimization은 이번 binary-addon의 필수 실행이 아니다.** 기본 `optimize_ellipsoid_lower=false`로 두어 새 benchmark 추가보다 큰 별도 최적화 작업을 자동으로 시작하지 마라. 위 analytic/grid bounds는 기존 계산에서 바로 기록한다. 이미 호환되는 개선된 oracle diagnostics가 있으면 재사용한다. 별도 요청으로 이 옵션을 켠 경우에만 `t=[0,31,63,95,127]`에서 모든 기본 registry method, 모든 10 interactions의 posterior를 freeze하여 다음 방식으로 lower bound를 개선한다.

- parameterization: `q_i=mu+rho_E W^(1/2) z_i`, each `||z_i||<=1`, i=1,2.
- 8-dimensional pair variables의 full region을 허용한다. direction/shape/center를 RC line으로 고정하지 않는다.
- 시작점은 feasible 7 grid queries, ±principal ellipsoid axis pair, 각 coordinate-axis pair, 그리고 deterministic fixed-RNG interior/boundary starts를 포함한다.
- current displayed pair는 E_t 안에 있을 때만 시작점에 포함한다.
- 기본 optimization budget은 8개 seeded random starts 추가, 각 start 최대 100 steps, 명시적 constraints, frozen common integration nodes; validated existing optimizer가 있으면 우선 재사용한다.
- 결과가 나쁘다고 특정 method/seed에만 restarts를 늘리지 마라. numerical failure에 대한 공통 refinement rule은 별도다.

로컬 최적화가 반환한 값을 true global maximum이라고 부르지 않는다. `eig_ellipsoid_lower`, `eig_ellipsoid_upper`, `oracle_gap`, `solver_status`, `posterior_numerical_target`을 저장한다. 비교 결과를 exact oracle point curve로 위장하지 마라.

selected EIG를 N_t라고 할 때 표시할 수 있는 것은 다음 구간이다.

\[
N_t/U_t\ \le\ \eta_{t,E}\ \le\ N_t/L_t.
\]

E_t 안의 selected query라면 오른쪽 endpoint를 1로 좁힐 수 있으나, E_t 밖이면 그렇게 하면 안 된다. L_t/U_t가 수치적으로 unresolved이면 flag한다.

이 bound가 유효한 대상은 **사용한 frozen posterior measure**다. finite-particle measure에서는 그 measure의 정확한 정보량에 대한 bound이며, 실제 continuous posterior까지 엄밀히 포함하려면 integration-error control이 더 필요하다. Monte Carlo 차이를 formal certificate라고 쓰지 않는다. continuous integration의 오차가 인증되지 않았다면 `numerical oracle bounds under posterior approximation`으로 표기한다.

기존 M=2 controls에 basic bound sidecar가 없으면 저장된 EIG/moments에서 먼저 계산한다. 충분한 정보가 없거나 별도 oracle-optimization 옵션을 켠 경우에만 저장된 actual history/checkpoint를 replay하되 **새 answers나 새 adaptive trajectory는 생성하지 않는다**. M>2는 이 작업에서 평가를 추가하지 않는다. 이론상 I*를 정의할 수 있는 것과 실험에서 global maximum을 수치적으로 구한 것은 구분한다.

### 7.4 Randomized-query diagnostics

PS, GC, SM, GCQ-1/3/5에 대해 t=[0,31,63,95,127]에서 64개의 독립 diagnostic pairs를 뽑아 conditional next-query-averaged EIG와 grid ratio를 저장한다. matched frozen oracle bounds가 있으면 mean EIG를 그 bounds로 나눈 구간도 저장한다.

Schedule state와 RNG를 clone한다. diagnostic pairs에 actual answers를 생성하지 않고, live posterior나 stage center를 바꾸지 않는다. 기본적으로 robot rollout은 하지 않는다. 이 conditional mean을 10개의 actual next-query ratio 평균과 혼동하지 않는다.

---

## 8. Meta-World: 기존 adapter를 그대로 재사용

기존 완료 실험은 **q-coordinate ideal-point oracle + simulator realization audit**이다. 이번 추가 실험도 동일하다. 로봇 dynamics나 actual trajectory features로 utility를 바꾸지 마라.

`reach-v3`, H=50, nominal sequence, 4 temporal motion bases, fixed gripper, shared context, global gain을 기존 serialized artifacts에서 로드한다. 현재 task로 성능을 개선하려고 새 controller/gain/context를 선택하지 마라.

원 설정은 Cartesian temporal basis `sin(pi*u)e_x`, `sin(pi*u)e_y`, `sin(pi*u)e_z`, `sin(2*pi*u)e_x`와 conservative q envelope 33을 쓴다. 새 방법의 displayed q도 이 envelope를 만족하는지 runtime 검증한다.

- MEM/GMEM/2MEM/PQ/QI의 지정 domain은 [-1,1]^4 안.
- IT half-spacing <=2.
- IB half-spacing <=10*sqrt(trace)<=20이며 physical cap도 검사.
- SM/GCQ의 displayed query는 current posterior mean 중심 radius32 cap.

q에 per-query normalization을 적용하지 마라. adapter bound를 위반하면 새 전체 gain으로 몰래 바꿔 과거와 비교하지 말고 실패를 보고한다.

모든 **새 selected candidate**를 같은 initial physical context에서 실제 실행한다. 기존 compatible rollout cache는 exact q bytes + context/adapter/simulator hash가 같을 때만 재사용한다.

저장해야 할 항목:

- displayed/raw q, actual submitted commands, command reconstruction error.
- selected logical rollout request의 executed/cache_hit/failed 상태.
- end-effector position time series, finite state/action, actual length.
- saturation, clipping, exception, terminal/truncation, task-success 보조 flag.
- pairwise RMS motion separation와 q-coordinate separation.

통계가 완료되었다고 physical replay까지 완료된 것으로 표시하지 마라. 실행되지 않은 clip을 executed라고 저장하지 마라. 실제 motion sensitivity/유효성 실패가 있더라도 statistical history를 삭제하지 않는다. simulator 실행이 collision-free나 real-robot safety를 증명했다고 쓰지 마라.

기존 calibration이 동일 adapter/context 버전에 대해 있으면 재사용한다. 신규 extreme q들의 고정된 preflight audit만 추가한다. RGB rendering은 quantitative hot path에서 하지 않는다.

---

## 9. 검증 tests: queue를 열기 전에 통과시킬 것

### 9.1 Scheduler exactness

- L=1/3/5의 stage lengths/starts가 Section 4.2와 정확히 일치.
- L3: t=41/42,83/84; L5: t=24/25,49/50,74/75,99/100 경계 검사.
- 각 boundary에서 이전 모든 답을 반영한 mean으로 한 번 recenter.
- 같은 stage의 proposals는 같은 center와 std 사용.
- std ratio=1/2, variance ratio=1/4; 이 둘을 혼동한 구현 탐지.
- T=128 vs T=4096의 stage schedule은 다른 config로 인식.
- original SG 함수 boundary/outputs 불변.
- deterministic centers와 Gaussian draws를 주입해 author-code schedule formula와 numerical output 대조.
- diagnostic/clone/resume가 stage나 actual RNG를 소비하지 않음.

### 9.2 Binary selector correctness

- 모든 새 selector는 shape(2,4), M!=2 reject.
- SM의 hyperplane tangent 보존/normal 반사, equal cap factors.
- MEM discrete global objective를 작은 exhaustive 문제와 비교.
- one-moving-center solver를 fine grid 및 모든 breakpoint evaluation과 대조.
- continuous uniform[-1,1] approximation에서 MEM centers≈(-1/2,+1/2), GMEM fixed center=0/free center≈±2/3.
- 2MEM의 순서 1,2,1,2와 각 step의 frozen objective nonincrease.
- IT에서 trace의 모든 dimensions 사용.
- IB의 raw/normalized Gaussian objective의 동치, entropy sign, bound enforcement, boundary optima.
- IB의 auxiliary Gaussian/confidence model이 actual likelihood에 침투하지 않음.

### 9.3 Shared numerical checks

- 기존 squared-distance likelihood, norm-offset, logsumexp tests.
- pair label permutation에 EIG 불변; identical candidates EIG=0.
- 실제 후보만으로 factorization 판단.
- actual EIG 평가가 pre-answer posterior에서 수행됨.
- grid EIG와 Fisher score는 별도; eta_grid<=1 검사는 RC에만.
- ellipsoid containment, grid lower bound, analytic upper bound, ratio interval 방향 검사.
- approximate oracle optimizer 결과를 true optimum이라고 표기하지 않음.
- posterior trace와 actual error의 분리, 초기 trace4/3.
- 동일 frozen history의 CPU/GPU 계산과 doubled-resolution 검사.

### 9.4 Integrity / robot / resume

- 기존 completed artifacts hashes 불변.
- 새 registry 순서와 worker 수가 seed mapping에 영향 없음.
- old/new 동일 interaction latent/context 값 일치.
- stage 직전/직후 checkpoint resume 일치.
- actual submitted command로부터 q 복원, context restore, candidate-order invariance.
- renderer/viewer가 hot loop에 생성되지 않음.
- physical GPU 3에 CUDA context/할당/계산이 없음.

---

## 10. 가장 빠른 incremental execution

### 10.1 자원 제한은 기존 서버 지시를 따른다

대상 서버는 기존 run에서 사용한 `acss-4090-server`다. 현재 접속 환경과 허용 자원을 실제 확인한다. 접속 권한 없이 실행했다고 보고하지 않는다.

```text
Allowed physical GPUs: 0,1,2 only
Forbidden compute GPU: 3
Expected CPU: AMD EPYC 9354, 32 physical / 64 logical
Expected RAM: 251 GiB
```

이것은 예상 자원이며 실측 affinity/cgroup/free memory와 다른 사용자의 점유를 확인한다. 다른 process 종료, GPU reset, driver/root 변경, paid compute는 하지 않는다.

초기화 전에 `CUDA_VISIBLE_DEVICES=0,1,2` 또는 동등한 isolation을 설정하고 physical-device mapping을 기록한다. GPU3에는 CUDA context를 만들거나 tensor를 할당하지 않는다. 읽기 전용 hardware inventory가 해당 device를 나열하는 것과 compute 사용은 구분한다.

### 10.2 Job 배치

- independent `(setting, method, interaction)` jobs에 대한 dynamic queue.
- GPU0/1/2당 persistent CUDA worker 하나로 joint SMC/large EIG batches 처리.
- CPU의 기존 marginal RC/PQ/QI 및 binary SM/MEM/IT/IB 작업과 rollout pool을 동시 실행.
- 단일 run에 DDP/DataParallel을 사용하지 않음.
- statistical producer는 robot rollout 완료를 매 round 기다리지 않음.
- GPU 경로가 float64에서 느리면 검증된 CPU 경로를 선택. 사용률 자체가 목표가 아님.

각 CPU worker의 BLAS/OpenMP/PyTorch threads는 우선 1로 설정한다. 가능한 범위에서 16/24/32 workers를 representative M=2 workload로 짧게 비교한다. 작업마다 environment를 새로 만들지 않는다.

이번에는 M16/beta16 workload를 다시 실행하여 profile할 필요가 없다. **추가되는 M=2 workload만** profile한다. 기존 profile도 호환되면 재사용한다. tuning은 correctness가 확보된 뒤 약 5–10분 안으로 제한한다.

### 10.3 가속 우선순위

1. compatible 완료 runs와 exact rollout cache 재사용.
2. marginal-factorization 가능한 binary methods를 정확하게 CPU에서 처리.
3. GCQ independent runs를 GPU batch화하고 frozen EIG 계산을 vectorize.
4. GPU particles/history coefficients를 device에 유지; full-history correction cache.
5. CPU rollout chunking, persistent contexts, low-overhead atomic checkpoints.
6. 마지막에 작은 subset만 render.

속도를 위해 particles=4096, Metropolis sweeps=10, 10 interactions, T128, selector definition을 바꾸지 마라. OOM에는 batch/concurrency만 줄인다.

### 10.4 Checkpoint / resume

16 answers마다와 stage boundary 이후 필요한 시점에 atomic checkpoint를 만든다. 다음을 포함한다.

```text
q/y history; posterior representation; RNG states;
stage number/center/std; cached stage proposals if precomputed;
inference adaptation; pending rollout IDs;
config/code/base-artifact hashes; metrics/status.
```

완료된 jobs는 skip, partial jobs는 마지막 valid checkpoint부터 resume한다. seed ID가 같아도 scientific config가 다르면 새 run이다. 같은 source schedule의 longer-horizon prefix라고 무조건 재사용하지 않는다.

---

## 11. 그림·표·보고서

### 11.1 새 M=2 outputs

최종 기본 14 configurations 모두에 대해 10 interactions를 병합한다.

- `binary_eta_grid_all`: pre-answer t=0,...,127의 realized eta_grid.
- `binary_posterior_risk_all`: t=0,...,128의 full posterior covariance trace.
- supplementary actual squared parameter error.
- per-seed traces, mean, endpoint tables: risk at answers32/64/128, eta at pre-answer31/63/127.
- `binary_schedule_audit`: GCQ1/3/5의 std와 stage-center changes; 기존 SG-legacy 결과가 있으면 별도 곡선으로 포함.
- `binary_ellipsoid_diagnostics`: fixed checkpoints의 oracle lower/upper 및 ratio intervals, numerical limitations.
- physical coverage/validity/separation tables 및 integration accuracy/runtime reports.

14 curves가 복잡하면 별도 grouped figures를 추가하되, 전체 method를 포함한 원본과 full numerical table도 제공한다. 그룹 예시는 controls, binary membership methods, finite-stage Gaussian schedules다. 결과 순위 때문에 특정 curve를 빼지 않는다.

기존 M>2 plot/data는 원본을 보존한다. 새 figure를 기존 M-sweep에 병합하고 싶으면 별도 파일을 생성하고 M=2 panel만 새 데이터로 바꾼다. M>2 numeric runs/rollouts를 재실행하지 않는다.

### 11.2 불확실성과 해석

기존 10-seed 보고 방식이 있으면 이어받는다. 없으면 individual curves + arithmetic mean을 기본으로 하고 표에 sample SD/SE와 n을 기록한다. 자동 p-value, 유의성 주장, ranking별 선택적 bootstrap/seed 추가는 하지 않는다. Finite T128로 asymptotic limits/rates를 입증했다고 쓰지 마라.

Method scope를 명시한다.

| 방법 | 이론/해석상의 주의 |
|---|---|
| PQ/PS/GC/QI 및 지정 SM/MEM/IT/IB | 원고의 exact-model 및 각 method별 조건하에 membership 논리를 사용; numerical approximation은 별도 |
| GCQ-1/3/5 | finite-stage empirical competitors; 자동 U_C 또는 vanishing-remainder membership 없음 |
| SG-legacy | 기존 결과 보존; T128에서 scale 변화 없음; infinite continuation과 finite-window 성능 구분 |
| MEM-line 계열 | coordinate-line specialization; arbitrary d-dimensional global optimization 아님 |
| IT/IB | component/bounded-relative transfers; full original Info-Synth 성능 비교라고 하지 않음 |

SG를 제외한 모든 published implementations가 무조건 U_C라는 문장을 만들지 마라. 각 theorem은 policy definition, posterior target, feasibility, cap 조건을 확인한 범위에만 적용한다.

### 11.3 Fidelity report

`PROVENANCE.md`에 source schedule과 transferred parts를 나눠 다음 표를 반드시 제공한다.

```text
method_id, paper/source locator, source version/hash,
supported M, objective, geometry, initialization,
std-vs-variance convention, stage allocation/rounding,
center estimator, posterior engine, catalog step,
cap, optimization budget, numerical approximation,
exact_reproduction_claim(false unless actually established).
```

GCQ의 source schedule unit test가 통과했다고 original catalog/ν-SVM implementation 전체를 검증했다고 쓰지 마라. MATLAB의 fixed Gaussian input과 Python schedule output 비교는 schedule translation check라고 부른다.

### 11.4 정성 영상

기존 renderer를 재사용한다. 사전 지정 subset: interaction ID 순서상 첫 interaction, t={0,63,127}, methods={RC,SM,GCQ-3}, 두 candidates. 최대18 selected clips이며 기존 compatible clip은 재사용한다. All numerical inference/replay 이후에만 render한다. 좋은 결과만 골라 subset을 바꾸지 않는다.

---

## 12. 설정 예시와 구현 산출물

아래 YAML은 구현 계약 예시다. 실제 registry/path에 맞춰 연결하고, 제공된 문자열을 이미 존재하는 module이라고 가정하지 마라.

```yaml
experiment:
  name: metaworld_m2_binary_addon_s10
  incremental_only: true
  base_run_manifest: AUTO_DISCOVER_AND_LOCK
  allowed_settings: [{id: m02_b04, dimension: 4, choices: 2, beta: 4.0}]
  answers_per_run: 128
  interaction_count: 10
  interaction_ids: INHERIT_EXACT_TEN_FROM_BASE
  theta_star: INHERIT_VALUES_FROM_BASE
  physical_contexts: INHERIT_MAPPING_FROM_BASE
  rerun_completed_controls: false
  rerun_m_gt_2: false
  add_beta_sweep: false
  modify_original_artifacts: false

controls:
  methods: [RC, PQ, QI, PS, GC]
  reuse_semantically_compatible: true
  run_only_missing_interactions: true
  legacy_sg: preserve_read_only_supplementary

new_methods:
  - sm_continuous
  - mem_line
  - gmem_line
  - 2mem_line_2sweeps
  - infosynth_trace
  - infosynth_bounded
  - gausscloud_paper_L1
  - gausscloud_paper_L3
  - gausscloud_paper_L5

gausscloud_paper:
  source_repo: siplab-gt/pairsearch
  source_commit: 9e621462af6b7ca502b362b1415c781dc10eb404
  source_path: enusvm/nusvm_sim.m
  source_blob_sha: 1969bfe73ca95027060b22ab0d2421cc446d7941
  stage_counts: [1, 3, 5]
  allocation: floor_each_remainder_to_final_stage
  std_shrink_factor: 0.5
  initial_scale_mapping: analytic_initial_prior_max_coordinate_sd
  initial_std: 0.5773502691896258
  center_mode: shared_posterior_mean_at_stage_entry
  intra_stage_recenter: false
  original_catalog_retrieval: false
  original_enusvm_estimator: false
  exact_end_to_end_reproduction: false
  cap: inherit_current_mean_radius_32

mem_family:
  geometry: coordinate_line
  center_domain: [-1.0, 1.0]
  marginal_solver: weighted_sorted_partition
  moving_center_solver: piecewise_quadratic_global
  two_mem_slot_order: [0, 1, 0, 1]
  until_convergence: false

infosynth_bounded:
  acquisition: auxiliary_gaussian_binary_conditional_entropy
  sigma0: 0.25
  relative_bounds: [0.01, 10.0]
  physical_halfspacing_cap: 32.0
  log_grid_points: 25
  sobol_points: 1024
  reference_sobol_points: 4096
  refinement_discrepancy_tolerance: 0.0005
  log_scale_tolerance: 0.00001
  replace_response_model: false

inference:
  inherit_existing_validated_backend: true
  dtype: float64
  initial_marginal_nodes: 257
  reference_marginal_nodes: 513
  smc_particles: 4096
  smc_reference_particles: 8192
  metropolis_sweeps_per_resampling: 10
  full_history_target: true
  factorization_check: actual_displayed_candidates

metrics:
  reuse_existing_eta_as: eta_grid
  main_comparison: [eta_grid, full_covariance_trace]
  save_realized_sq_error: true
  new_oracle_kind: inherit_latest_ellipsoid_do_not_rollback
  oracle_eval_default_alpha_E: 0.5
  oracle_eval_default_rho_E: 7.464101615137754
  oracle_bounds_each_round: true
  optimize_ellipsoid_lower: false  # separate opt-in diagnostic, not part of addon completion
  oracle_lower_optimization_t: [0, 31, 63, 95, 127]
  oracle_lower_optimization_interactions: all_ten
  extra_random_oracle_starts: 8
  max_steps_per_oracle_start: 100
  report_local_optimum_as_global: false
  randomized_query_draws: 64
  randomized_query_t: [0, 31, 63, 95, 127]
  diagnostics_mutate_live_state: false

robot:
  task: inherit_reach_v3
  clip_action_steps: 50
  adapter_and_gain: inherit_exact_serialized_version
  rollout_all_new_selected_candidates: true
  reuse_exact_cache: true
  approximate_q_cache: false
  render_in_hot_path: false
  reward_oracle_depends_on_trajectory: false

resources:
  allowed_physical_gpus: [0, 1, 2]
  forbidden_compute_gpus: [3]
  persistent_gpu_workers: one_per_allowed_gpu
  dynamic_queue: true
  cpu_worker_candidates: [16, 24, 32]
  cpu_worker_blas_threads: 1
  representative_profile: new_M2_workload_only
  change_scientific_budget_for_speed: false

output:
  root: NEW_ADDON_DIRECTORY
  checkpoint_every_answers: 16
  atomic_writes: true
  keep_base_hash_inventory: true
  save_preanswer_and_postanswer_metrics_separately: true
  save_raw_displayed_queries: true
  uncertainty: inherit_or_ten_individual_traces_plus_mean
  automatic_pvalues: false
  plot_formats: [pdf, png]
```

구현할 기능은 inspect/plan/test/profile/run/resume/validate/plot/report다. 기존 CLI에 subcommand를 붙여도 되고 별도 addon package를 만들어도 된다. 예를 들어 `metaworld_binary_addon`을 새로 만들었다면 실제 검증한 다음 형태의 실행 방법을 보고할 수 있다.

```bash
python -m metaworld_binary_addon.cli inspect --config <actual_config>
python -m metaworld_binary_addon.cli test --config <actual_config>
CUDA_VISIBLE_DEVICES=0,1,2 python -m metaworld_binary_addon.cli run --config <actual_config> --resume
python -m metaworld_binary_addon.cli validate --run-dir <actual_addon_run>
python -m metaworld_binary_addon.cli report --run-dir <actual_addon_run>
```

이 명령은 **새 CLI의 예시**다. module이 없는데 실행 가능하다고 주장하지 마라. 실제 생성·검증한 명령을 최종 보고서에 넣는다.

필수 산출물:

```text
BASELINE_LOCK.json
REUSE_PLAN.json
METHOD_SOURCE_AUDIT.md
PROVENANCE.md
CONFIG_DIFF.md
frozen_config.yaml
manifest.json
statistical/ + checkpoints/
oracle_sidecars/ + randomized_diagnostics/
rollout_requests/ + rollouts/
validation/ + tests/
figures/ + tables/ + selected_videos/
PROFILE.json + ETA.md + REPORT.md
```

---

## 13. 완료 판정

아래를 실제 artifacts로 확인한다.

1. 하나의 setting: d4, M2, beta4, T128, 동일한 기존 10 interactions.
2. 기본 14 configurations의 완료/재사용/실패 상태가 전부 명시됨.
3. controls 5개는 compatible 완료 run을 재실행하지 않음; M>2 runs와 결과는 불변.
4. 새 binary6 방법과 GCQ1/3/5가 모두 정확한 registry 이름으로 존재.
5. source-code scheduling tests, numerical comparisons, RNG/resume tests 실행 기록.
6. 모든 신규 selected candidates의 physical replay/cache/failure 상태와 actual motion diagnostics.
7. eta_grid와 ellipsoidal oracle/bounds를 이름과 정의에서 명확히 분리.
8. SG finite-stage baseline에 잘못된 U_C/rate claim이 없음.
9. 결과가 예상과 달라도 보존; seed/schedule/initial scale을 사후 조정하지 않음.
10. 새 계산·기존 재사용·frozen-history replay·미완료/uncertainty를 REPORT에서 분리.

**실행 순서:** base lock → source/selector audit → missing-job plan → correctness tests → 짧은 M2 profiling → frozen manifest → 신규 inference와 CPU rollout 병렬 실행 → fixed-history 검증/basic oracle-bound sidecars → figures/tables/report. 선택적 full-ellipsoid optimization은 별도 요청이 있을 때만 실행한다.

---

## 14. 직접 확인한 primary sources와 locator

이 절은 구현·provenance용이다. 기존 대화의 설명을 원문 검증 대신 사용하지 마라. URL은 실행자가 확인할 원본 위치이며, 원본이 바뀌면 pinned version을 유지한다.

### [BASE] 제공된 Meta-World 명세와 RC-MLQ 원고

- 제공 파일: `붙여넣은 마크다운.md`, 제목 `Codex 실행 명세: RC-MLQ × Meta-World 고속 실험`.
- 핵심 위치: Sections1.4,2,5–8,17, 마지막 ACSS4090 hardware override.
- 최신 사용자 override: 완료 실험은 10 interactions이며 M2만 추가.
- Binary definitions: 제공 RC-MLQ 원고 `붙여넣은 텍스트(1)(10).txt`의 `app:synthetic-policies`, `app:mem-rules`, `app:trace-scale-membership`, `app:gaussian-membership`.
- 수정 이론 파일이 repository에 있으면 `rcmlq_ellipsoidal_oracle.tex`의 `eq:ellipsoid-shape`, `eq:ellipsoid-radius`, `eq:ellipsoid-oracle`, `eq:grid-information-ratios`, `app:ellipsoid-evaluation`을 대조한다. 이 채팅 sandbox 경로가 원격 repository에 존재한다고 가정하지 않는다.

### [G-paper] Canal et al., ICML2019

Gregory Canal, Andy Massimino, Mark Davenport, Christopher Rozell. *Active Embedding Search via Noisy Paired Comparisons*. PMLR97,902–911,2019.

- `https://proceedings.mlr.press/v97/canal19a.html`
- `https://proceedings.mlr.press/v97/canal19a/canal19a.pdf`
- Section4.1 GaussCloud-Q description; Figure2 contains GaussCloud-1,3,5 and an80-query experiment.
- paper length80 is not copied into this experiment: user requires128.

### [G-code] Author code — implementation-level scheduling authority

- `https://github.com/siplab-gt/pairsearch/blob/9e621462af6b7ca502b362b1415c781dc10eb404/enusvm/nusvm_sim.m`
- Locator variable names: `sig`, `measperstage`, `measperstage(end)`, `sigstage`, `Z`, `xhat`, `enusvm_solve`.
- Source initialization is maximum coordinate SD of the fixed embedding, remainder goes to final stage, std halves each stage. Stage estimator/catalog procedures differ from this shared-Bayes continuous task.

### [G-theory] Massimino & Davenport, JMLR2021

Andrew K. Massimino, Mark A. Davenport. *As You Like It: Localization via Paired Comparisons*. JMLR22(186),1–39,2021.

- `https://jmlr.org/papers/volume22/18-105/18-105.pdf`
- Section5.2/Proposition10: staged Gaussian localization and equal stage budgets in the analysis.
- Sections6.3/6.4 use different experiment-specific rounding descriptions; Section6.4 discusses dyadic variance reduction. Do not erase those distinctions. This addon's implementation anchor is [G-code], not a hybrid of inconsistent conventions.

### [SM] Chumbalov et al., ICML2020

Daniyar Chumbalov, Lucas Maystre, Matthias Grossglauser. *Scalable and Efficient Comparison-based Search without Features*. PMLR119,1995–2005,2020.

- `https://proceedings.mlr.press/v119/chumbalov20a/chumbalov20a.pdf`
- Section4.1 Algorithm1: Gaussian sample, principal-hyperplane reflection, catalog retrieval.
- Section4.2: approximate Gaussian belief update in the source, not this experiment's posterior target.

### [MEM] Price & Messinger, AAAI2005

Bob Price, Paul R. Messinger. *Optimal Recommendation Sets: Covering Uncertainty over User Preferences*. AAAI2005,541–548.

- `https://cdn.aaai.org/AAAI/2005/AAAI05-085.pdf`
- Expected-maximum-utility criterion; Equation7 GMEM and Equation8 local reoptimization.
- Original2MEM iterates while improving; current manuscript explicitly chooses two sweeps. Original methods are not intrinsically binary; this addon implements the manuscript's binary coordinate specialization.

### [IS] Nadagouda et al., inspected2026 preprint

Namrata Nadagouda, Nauman Ahad, Maegan Tucker, Mark A. Davenport. *Active Query Synthesis for Preference Learning*. arXiv2605.26072v1.

- `https://arxiv.org/html/2605.26072v1`
- Section4.2 and AppendixB.3: symmetric-pair acquisition and trace-scale approximation. The inspected work is a preprint; do not invent a conference status.
- IT/IB are the source-component/bounded transfers specified in the provided manuscript, not the entire original method.

**마지막 지시:** 위 scientific scope를 유지하면서 가장 빠른 검증된 경로로 실행하라. SG/GCQ가 RC보다 잘 나오는 결과도 그대로 보존하라. 원본에서 가져온 부분, 현재 설정으로 이식한 부분, 실제 실행한 부분을 분리하여 보고하라.
