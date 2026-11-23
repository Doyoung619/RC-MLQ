# 앞으로 사용하는 합성 실험 알고리즘 세트

사용자 요청(2026-10-05)에 따라 EmpiricalExperiment ZIP의 구현 명세와 실제
source snapshot을 SyntheticExperiment에 연결했다. 원문은
`ALGORITHM_IMPLEMENTATION_SPEC.md`, 원본 소스·ZIP의 SHA256은
`algorithm_source_manifest.json`에 보존했다. 수치 커널 selectors.py,
family_selectors.py, numerics.py와 독립 replay는 원본 그대로 복사했다.
외부 EmpiricalExperiment 폴더 없이 SyntheticExperiment만으로 실행된다.

## 방법과 표시

| 내부 ID | 표시 | 적용 모델 |
|---|---|---|
| fb_opt2 | Ours | 공통 |
| a_opt | A-opt | 공통 |
| d_opt | D-opt | 공통 |
| e_greedy | E-greedy | 공통 |
| e_sdp | E-SDP | 공통 |
| bait_glm | BAIT | 공통 |
| ucb_glm | UCB | 공통 |
| ts_glm | TS | 공통 |
| random | Random | 공통 |
| b_emcm | B-EMCM | Gaussian |
| lcmd | LCMD | Gaussian |
| regvol | RegVol | Gaussian |
| entropy | Entropy | Bernoulli/logistic |
| badge | BADGE | Bernoulli/logistic |
| firal | FIRAL | Bernoulli/logistic |

각 모델에 12개, 합쳐서 고유 알고리즘 15개다. 설정에서 methods를 생략하면
이 목록 전체를 모델 적합성에 맞게 확장한다. `ours`는 `fb_opt2`의 별칭이다.
**Ours는 #0057B8 파란색 실선**, 범례 첫 항목이다. 번들의 빨간색/diamond
그림 설정 대신 사용자가 지정한 스타일을 적용한다.

## 실행 설정

`configs/synthetic_main.yaml`이 앞으로의 기본 합성 campaign이다.

- d=5,10,20; m=20,40,80; seed=0,...,9.
- Gaussian 분산 1, 추가 예산 400; Bernoulli 추가 예산 4000.
- 초기 관측 80, Gaussian 후보 cap 256, Bernoulli cap 512.
- 총 2160개 job. 초기 관측은 추가 예산과 온도 sample index에서 제외한다.
- 최대 20차원 anisotropic pool을 생성하고 d별로 앞 d개 열을 사용한다.
  최대 차원 query로 초기 D-opt 80개를 정하므로 동일 seed에서 d/m/method
  모두 같은 초기 ID를 공유한다. full pool은 6000개 생성 row + 40개 축 anchor.
- 기존 합성 generator의 단위 구 geometry와 알려진 theta_star를 사용한다.
  theta_regime=weak, signal_std=0.8, theta_norm_cap=8. 절편은 새로 붙이지 않는다.
  좌표별 [-3,3] 추정 제약은 없다. 파라미터 추정은 unconstrained ridge MLE다.
- 명세의 without-replacement 규칙을 적용한다. 동일 feature를 가진 다른
  row는 별개 ID지만 이미 관측한 ID를 다시 선택할 수 없다.
- seed마다 합성 pool/정답/응답이 생성되며 같은 seed/d/family의 모든 방법은
  이를 공유한다. empirical 번들의 고정 실제 응답 실험과 구분해야 한다.
- Ours: tau_range=max(1e-8,R*sqrt(m/(8*log(max(2,d)))))이고
  tau_t=tau_range*(m*t/Nmax)^0.75. R은 Gaussian 1, Bernoulli 1/4,
  Nmax는 설정의 추가 예산. 마지막 tau는 정확히 tau_range이다.
  pilot/smoke는 명시적으로 더 짧은 계획 예산을 사용한다.
- UCB/TS alpha=1; B-EMCM Z=4, virtual alpha=0.05; RegVol lambda=1;
  FIRAL eta=[2,20,200,2000,20000], normalized minimum-eigenvalue로 선택.
- Gaussian/Bernoulli만 이 세트를 지원한다. Poisson과 과거 알고리즘은
  기존 명시적 historical config 경로로 남아 있다.

과거 `synthetic_main.yaml`, `synthetic_pilot.yaml`, `smoke.yaml`은
`configs/legacy`에 원본을 보관했다. 기존 asymptotic_preview의 d=6,
반복 질의, tau0=0.02 설정은 새 FB-opt2 campaign과 다르다.

## 선택·학습·수치 검증

`spec_suite.select_batch`가 명세의 label 제한 인터페이스다. B-EMCM만
누적 observed_y를 받으며 hidden y_pool, theta_star, evaluation을 받지 않는다.
배치 전체 선택 후 신규 응답을 공개하고 누적 데이터로 다시 적합한다.
매 round Fisher를 현재 theta에서 재계산한다. Bernoulli 공통 acquisition
floor는 1e-12이고 평가/BAIT target/FIRAL에는 floor를 넣지 않는다.

sum loss + rho||theta||²/2, rho=1이며 기존 절편이 있으면 같이 penalize한다.
Gaussian Cholesky, Bernoulli damped Newton/동일 목적함수 BFGS retry/Newton polish를
사용한다. predictor clipping을 하지 않고 scaled gradient <=1e-8을 확인한다.
수렴하지 않은 결과는 실패로 기록한다. E-SDP는 SCS/strict SCS/CLARABEL gate,
dependent rounding 8회, E-greedy와 비교 및 one-swap을 보존한다.
실패한 SDP만 audit에 이유를 남기고 E-greedy로 fallback한다.

selector seed는 SHA256('|'.join(('selector',dataset,d,seed,internal_method)))의
첫 8바이트 little endian이다. candidate/tie/initialization priority는
SHA256(repr(tuple))와 SplitMix64를 사용한다. trajectory 동안 RNG를 이어 쓴다.

## 저장·재개·평가

`results_spec/<experiment>/<run_id>/checkpoints/round_XXXXXX.npz`와 `.json`에
각 completed batch를 atomic commit한다. NPZ에는 관측 prefix, 관측 응답,
theta_hat, signed_error, exact true/plugin Fisher와 spectra를 저장한다.
JSON에는 후보/선택 ID, frozen theta, RNG 상태, solver/fitting 진단과 hash가 있다.
수치 방법별 bootstrap, reverse-removal, BADGE embedding/D²,
FIRAL relaxation/eta 정보를 selector audit 그대로 보존한다.

연속 JSON commit과 일치하는 NPZ hash만 resume한다. RNG도 복원한다.
미완료 마지막 NPZ는 commit 전 상태이므로 이전 completed batch부터 재계산한다.
중간 commit 손상/누락이나 다른 입력·코드 hash는 조용히 건너뛰지 않고 실패한다.
완료 run도 최종 파일 및 체크포인트 hash를 검사하고 재사용한다.

평가 namespace는 선택기로 전달하지 않는다. 최종 trajectory.csv에는
l2_error, squared_error, lambda_min(true), plugin_lambda_min을 저장한다.
report는 seed별 signed error의 **uncentered** Q=mean(e e.T)에서
R2=sqrt(trace(Q)), Rdir=sqrt(lambda_max(Q))를 계산한다. 정보량은
seed별 최소 고유값을 먼저 계산한 뒤 평균한다. 전체 평균 행렬의 최소
고유값이나 평균 error norm으로 대체하지 않는다.

새 그림은 true information / plugin information / R2 / Rdir의 2x2이며
x축은 추가 응답 수 mT, 선형 축, smoothing과 confidence shading은 없다.
CSV에는 정보량의 descriptive Student-t interval을, endpoint에는 whole-seed
1000회 bootstrap risk 구간을 저장한다. leave-one-seed-out 진단도 저장한다.
출력은 paper/generated/algorithm_suite 아래 PDF/PNG/SVG와 CSV다.

## 명령

아래 명령은 SyntheticExperiment 폴더에서 실행한다.

```powershell
python -m pip install -e ".[dev]" --no-build-isolation
python -m pscore_harness plan --config configs/smoke.yaml
python -m pscore_harness run --config configs/smoke.yaml --workers 1 --fail-fast
python -m pytest -q
python -m pscore_harness validate --config configs/smoke.yaml
python -m pscore_harness report --results-root results_spec/smoke_algorithm_suite --paper-dir paper/suite_smoke
```

pilot는 configs/synthetic_pilot.yaml, 전체 실험은 configs/synthetic_main.yaml로
동일하게 plan/run/validate한다. 병렬 worker 수는 machine 자원에 맞추고
각 worker의 BLAS thread는 1로 둔다. 구현 요청 시 전체 campaign은 실행하지
않으며 smoke와 테스트로 구현을 검증한다.

## 이번 구현 검증

`algorithm_suite_validation.json`에 검증 결과를 저장했다. 테스트 63개 통과,
2-worker smoke 24개 job 및 72개 fit checkpoint 모두 완료했다. 미수렴 fit과
E-SDP fallback은 0개였고 FIRAL의 최대 relative gap은 7.1327334e-8이었다.
중단 후 stochastic RNG와 추정값의 정확한 재개, 손상 checkpoint 감지,
독립 수식·분포 검사를 포함한다. Gaussian/Bernoulli 검증 그림의 12개 범례와
Ours 파란색 실선을 시각적으로 확인했다. 전체 2160-job campaign은 실행하지 않았다.
