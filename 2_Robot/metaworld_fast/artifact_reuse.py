"""Record explicit eligibility checks; do not relabel incompatible histories."""
from pathlib import Path

def inspect_reuse():
    return {
        'reused_statistical_runs': 0,
        'reason': 'Ideal_Control contained only the supplied execution specification. No compatible q/y history artifacts were found.',
        'inspected_implementation': '/data1/doyoung/shared',
        'incompatibilities': ['Gaussian prior rather than bounded uniform prior',
                              'Laplace Gaussian posterior rather than marginal integration / tempered SMC',
                              'Gaussian quantile query template and different scales'],
        'manuscript_available': (Path(__file__).parent.parent / 'Ideal_Type_Finder(3).pdf').exists(),
        'definition_source': 'CODEX_RCMLQ_METAWORLD_FAST_T128_S5_KO.md',
    }
