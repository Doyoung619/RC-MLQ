# 그림용 정리 데이터

저장소에는 그림을 다시 그리는 데 필요한 데이터와 plotting 코드만 포함합니다.

| 파일 | 내용 |
| --- | --- |
| `figure_editing/data/curves.csv.gz` | 원래 24종 figure에 쓰인 개별 곡선 CSV, gzip 약 2.7 MiB |
| [plot_figures.py](figure_editing/plot_figures.py) | gzip CSV를 직접 읽어 PDF/SVG/PNG 생성 |
| [style.json](figure_editing/style.json) | 색상, 라벨, 선 굵기, 축 범위 |
| [figure_index.csv](figure_editing/figure_index.csv) | 24종 figure와 metric 목록 |
| [사용법.txt](figure_editing/사용법.txt) | 실행 명령과 데이터 열 설명 |
| [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json) | 압축 파일과 원본 CSV의 크기·SHA-256 |

CSV는 65,790행이며 7개 setting, interaction ID 0~9의 곡선을 담습니다. 원래 CSV를 그대로 압축했으므로 값·정밀도·빈 셀은 동일합니다. 평균만 남기거나 수치를 반올림하지 않았습니다.

```bash
python -m pip install -r results/figure_editing/requirements.txt
python results/figure_editing/plot_figures.py --list
python results/figure_editing/plot_figures.py --formats svg pdf
```

그림은 `figure_editing/edited_figures/`에 생성되며 Git에서 제외합니다. GPU, MuJoCo, checkpoint 없이 실행할 수 있습니다.

## 해석과 원본

Ellipsoid 정보비율의 분모는 달성한 수치 EIG이며 exact global optimum을 인증하지 않습니다. 기존 posterior·integration·optimization 오차는 그대로 남아 있습니다. M=2 후속 검증에서 PS·GC·GCQ-1/3/5의 50개 실행은 수치 정확도 기준 미달로 기록되었습니다. [LIMITATIONS.md](../metaworld_fast/LIMITATIONS.md)와 [PROVENANCE.md](../metaworld_fast/PROVENANCE.md)도 참고하세요.

전체 실험 원본은 로컬 `metaworld_fast/outputs/`, `metaworld_fast/engineering/`, `exports/`에 보존합니다. 이전에 만든 `results/runs/` 사본과 완성된 figure·robot preview도 로컬에 남기고 Git 추적에서 제외했습니다. 상세 run 데이터·검증 JSON·checkpoint·rollout·중복 그림은 업로드하지 않습니다.
