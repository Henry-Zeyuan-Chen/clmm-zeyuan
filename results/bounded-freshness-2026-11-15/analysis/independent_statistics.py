"""Fixed-checkpoint history bootstrap from independently audited score vectors."""
import numpy as np

from independent_tools import ARMS, PARENTS, ROUTES, aggregate_histories, bootstrap_ids, interval


def summarize(vectors, all_correct, *, n=128, draws=5000, seed=91204):
    ids = bootstrap_ids(n, draws, seed)
    assert n % 16 == 0
    block_ids = bootstrap_ids(n//16, draws, seed)
    expected = {(parent, route) for parent in PARENTS for route in ROUTES}
    assert {(p, r) for p, r, *_ in vectors} == expected
    gates, aggregate, contrasts, per_condition = [], [], [], []
    for parent, route in sorted(expected):
        ac = lambda cell, arm='online': np.asarray(all_correct[(parent, route, cell, arm)], dtype=float)
        rolled = ac('acquire2')-ac('acquire2', 'rolled')
        rolled_ci = interval(rolled.reshape(n//16, 16).mean(axis=1), block_ids)
        acquire2, acquire4, revise2 = (float(ac(cell).mean()) for cell in ('acquire2', 'acquire4', 'revise2'))
        short = acquire2 >= .9 and acquire4 >= .8 and revise2 >= .9 and rolled_ci['lower'] > 0
        neutral = float(ac('gate_k16_r3').mean())
        common = bool(short and neutral >= .8)
        for arm in ARMS:
            for phase in (0, 2):
                d0 = float(ac(f'pressure_k16_p{phase}_d0', arm).mean())
                gates.append(dict(parent_seed=parent, route=route, arm=arm, phase=phase,
                                  acquire2_all_correct=acquire2, acquire4_all_correct=acquire4,
                                  revise2_all_correct=revise2, short_gate=bool(short),
                                  acquire2_online_minus_rolled=rolled_ci['estimate'],
                                  rolled_lower95=rolled_ci['lower'], rolled_upper95=rolled_ci['upper'],
                                  rolled_bootstrap_unit='batch16 because rolled states share histories',
                                  neutral_six_target_all_correct=neutral,
                                  common_checkpoint_admission=common, own_D0_six_target_all_correct=d0,
                                  own_full_retention_gate=bool(common and d0 >= .8),
                                  included_in_primary=True))

    for family in ('balanced', 'content', 'mean5random'):
        for phase in ('mean', 0, 2):
            phases = (0, 2) if phase == 'mean' else (phase,)
            for tail in (0, 64, 128):
                cells = tuple(f'pressure_k16_p{p}_d{tail}' for p in phases)
                categories = ('stable', 'revised', 'early', 'eligible', 'targets') + (('recent',) if tail else ())
                for category in categories:
                    av = {arm: aggregate_histories(vectors, cells, arm, category, family) for arm in ARMS}
                    for arm in ARMS:
                        ci = interval(av[arm], ids)
                        aggregate.append(dict(route_family=family, phase=phase, retention_tail=tail,
                                              arm=arm, category=category, accuracy=ci['estimate'],
                                              lower95=ci['lower'], upper95=ci['upper'], histories=n))
                    literal = (av['continuous']+av['periodic'])/2
                    bounded = (av['continuous_bounded']+av['periodic_bounded'])/2
                    comparisons = [
                        ('bounded', 'literal', bounded, literal, 'mean', 'freshness'),
                        ('continuous_bounded', 'continuous', av['continuous_bounded'], av['continuous'], 'continuous', 'freshness'),
                        ('periodic_bounded', 'periodic', av['periodic_bounded'], av['periodic'], 'periodic', 'freshness'),
                        ('periodic', 'continuous', av['periodic'], av['continuous'], 'literal', 'timing'),
                        ('periodic_bounded', 'continuous_bounded', av['periodic_bounded'], av['continuous_bounded'], 'bounded', 'timing'),
                        ('periodic_freshness', 'continuous_freshness', av['periodic_bounded']-av['periodic'], av['continuous_bounded']-av['continuous'], 'interaction', 'timing_x_freshness'),
                    ]
                    for arm in ARMS[2:]:
                        comparisons.extend((arm, base, av[arm], av[base], arm, 'anchor') for base in ('repeat', 'online'))
                    comparisons.append(('repeat', 'online', av['repeat'], av['online'], 'anchor', 'anchor'))
                    for candidate, baseline, cv, bv, schedule, contrast_kind in comparisons:
                        primary = (family == 'balanced' and phase == 'mean' and tail == 128
                                   and category in ('stable', 'revised') and schedule == 'mean'
                                   and contrast_kind == 'freshness')
                        ci = interval(cv-bv, ids, level=.975 if primary else .95)
                        contrasts.append(dict(route_family=family, phase=phase, retention_tail=tail,
                                              category=category, candidate=candidate, baseline=baseline,
                                              schedule=schedule, contrast_kind=contrast_kind,
                                              candidate_accuracy=float(cv.mean()), baseline_accuracy=float(bv.mean()),
                                              **ci, role='co-primary' if primary else 'secondary',
                                              histories=n, bootstrap_draws=draws, bootstrap_seed=seed))
    for (parent, route, cell, arm, category), value in sorted(vectors.items()):
        if cell.startswith('pressure_'):
            ci = interval(value, ids)
            per_condition.append(dict(parent_seed=parent, route=route, cell=cell, arm=arm,
                                      category=category, accuracy=ci['estimate'],
                                      lower95=ci['lower'], upper95=ci['upper']))
    primary = [r for r in contrasts if r['role'] == 'co-primary']
    assert len(primary) == 2
    relevant = [r for r in gates if r['arm'] in ARMS[2:]]
    retention = all(r['own_full_retention_gate'] for r in relevant)
    return dict(gates=gates, aggregate=aggregate, contrasts=contrasts, per_condition=per_condition,
                primary=primary, all_common_checkpoint_gates_pass=all(r['common_checkpoint_admission'] for r in gates),
                all_replay_retention_gates_pass=retention,
                interpretation='admitted retention comparison' if retention else 'revision/interference diagnostic; all conditions retained',
                bootstrap=dict(seed=seed, draws=draws, histories=n, co_primary_interval_level=.975,
                               secondary_interval_level=.95, co_primary_quantiles=[.0125, .9875],
                               paired_unit='whole history, jointly across phases/delays/arms/parents/maps',
                               conditioning='two fixed learned initializers, one fixed content map, all five fixed random maps',
                               balanced_weighting='.5 content + .5 mean(all5random); equal average2parents/2phases/2schedules',
                               scope='conditional history uncertainty; not population training or routing-map uncertainty'))
