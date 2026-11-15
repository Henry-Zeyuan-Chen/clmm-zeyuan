"""Fixed-parent/map paired-history contrasts with explicit joint qualification."""
import numpy as np

SEEDS = (90511,90515)
ROUTES = ('content',*(f'random-{s}' for s in range(90701,90706)))


def summarize(cases,draw_seed=90804,draws=5000,margin=.03):
    """cases[(rule,parent_seed,route)] has independently audited vectors/all_correct."""
    n = len(next(iter(cases.values()))['vectors'][('revise2','online','revised')])
    assert n % 16 == 0
    ids = np.random.Generator(np.random.PCG64(draw_seed)).integers(0,n,size=(draws,n))
    blocks = np.random.Generator(np.random.PCG64(draw_seed)).integers(0,n//16,size=(draws,n//16))
    rules = sorted({key[0] for key in cases})
    assert set(cases) == {(rule,s,r) for rule in rules for s in SEEDS for r in ROUTES}
    gates,gate_lookup = [],{}
    for (rule,s,route),data in sorted(cases.items()):
        ac = data['all_correct']
        values = {cell:float(ac[(cell,'online')].mean()) for cell in ('acquire2','acquire4','revise2')}
        rolled = ac[('acquire2','online')]-ac[('acquire2','rolled')]
        lo,hi = np.quantile(rolled.reshape(n//16,16).mean(1)[blocks].mean(1),[.025,.975])
        acquisition = bool(values['acquire2'] >= .9 and values['acquire4'] >= .8 and lo > 0)
        short = bool(acquisition and values['revise2'] >= .9)
        row = dict(rule=rule,parent_seed=s,route=route,**values,
                   normal_minus_rolled=float(rolled.mean()),rolled_lower95=float(lo),rolled_upper95=float(hi),
                   acquisition_gate=acquisition,full_short_gate=short,
                   reset_acquire2=float(ac[('acquire2','reset')].mean()))
        for k in (8,16):
            d0 = float(ac[(f'pressure_k{k}_d0','online')].mean())
            row[f'K{k}_D0_four_original_all_correct'] = d0
            row[f'K{k}_D0_gate'] = bool(d0 >= .8)
            row[f'K{k}_full_admission'] = bool(short and d0 >= .8)
            row[f'K{k}_baseline_repair_admission'] = bool(acquisition and d0 >= .8)
        gate_lookup[(rule,s,route)] = row
        gates.append(row)
    def aggregate(rule,routes,cell,category):
        return np.mean([cases[(rule,s,r)]['vectors'][(cell,'online',category)]
                        for s in SEEDS for r in routes],axis=0)
    contrasts,aggregate_rows,permap = [],[],[]
    for family,routes in [('content',('content',)),('mean5random',ROUTES[1:])]:
        candidate_pass = all(gate_lookup[('local_delta',s,r)]['K8_full_admission'] for s in SEEDS for r in routes)
        baseline_repair_pass = all(gate_lookup[('ema',s,r)]['K8_baseline_repair_admission'] for s in SEEDS for r in routes)
        baseline_full_pass = all(gate_lookup[('ema',s,r)]['K8_full_admission'] for s in SEEDS for r in routes)
        for cell,category,label,threshold in [('revise2','revised','revision_superiority',0.),
                                               ('pressure_k8_d64','stable','stable_noninferiority',-margin)]:
            candidate = aggregate('local_delta',routes,cell,category)
            baseline = aggregate('ema',routes,cell,category)
            delta = candidate-baseline
            lo,hi = np.quantile(delta[ids].mean(1),[.0125,.9875])
            contrasts.append(dict(route_family=family,contrast=label,cell=cell,category=category,
                                  candidate=float(candidate.mean()),baseline=float(baseline.mean()),
                                  estimate=float(delta.mean()),lower=float(lo),upper=float(hi),interval_level=.975,
                                  lower_tail_probability=.0125,threshold=threshold,inequality_pass=bool(lo>threshold),
                                  candidate_all_required_gates=candidate_pass,baseline_repair_required_gates=baseline_repair_pass,
                                  baseline_original_full_admission=baseline_full_pass,
                                  admitted_inequality_pass=bool(lo>threshold and candidate_pass and baseline_repair_pass),
                                  histories=n,bootstrap_draws=draws,bootstrap_seed=draw_seed,
                                  scope='Bonferroni four one-sided claims; fixed2parents/5randommaps; history-cluster paired bootstrap'))
        for rule in rules:
            for cell,arm,category in cases[(rule,SEEDS[0],routes[0])]['vectors']:
                if arm != 'online' or category == 'targets':
                    continue
                v = aggregate(rule,routes,cell,category)
                lo,hi = np.quantile(v[ids].mean(1),[.025,.975])
                aggregate_rows.append(dict(rule=rule,route_family=family,cell=cell,category=category,
                                           accuracy=float(v.mean()),lower95=float(lo),upper95=float(hi)))
        if set(rules) == {'ema','additive','erase_ema','local_delta'}:
            for cell,category in [('revise2','revised'),('pressure_k8_d64','stable')]:
                v = {r:aggregate(r,routes,cell,category) for r in rules}
                controls = dict(erasure_with_bulk_decay=v['erase_ema']-v['ema'],
                                erasure_without_bulk_decay=v['local_delta']-v['additive'],
                                bulk_decay_without_erasure=v['ema']-v['additive'],
                                bulk_decay_with_erasure=v['erase_ema']-v['local_delta'],
                                erasure_by_bulk_decay_interaction=(v['erase_ema']-v['ema'])-(v['local_delta']-v['additive']))
                for label,delta in controls.items():
                    lo,hi = np.quantile(delta[ids].mean(1),[.025,.975])
                    contrasts.append(dict(route_family=family,contrast=label,cell=cell,category=category,
                                          candidate='',baseline='',estimate=float(delta.mean()),
                                          lower=float(lo),upper=float(hi),interval_level=.95,lower_tail_probability=.025,threshold='',
                                          inequality_pass='',candidate_all_required_gates='',baseline_repair_required_gates='',
                                          baseline_original_full_admission='',admitted_inequality_pass='',
                                          histories=n,bootstrap_draws=draws,bootstrap_seed=draw_seed,
                                          scope='Factorial mechanism diagnostic; nominal95 interval'))
    for (rule,s,route),data in sorted(cases.items()):
        for cell,category in [('revise2','revised'),('revise2','stable'),('pressure_k8_d64','stable'),('pressure_k8_d64','revised')]:
            v = data['vectors'][(cell,'online',category)]
            old = cases[('ema',s,route)]['vectors'][(cell,'online',category)]
            permap.append(dict(rule=rule,parent_seed=s,route=route,cell=cell,category=category,
                               accuracy=float(v.mean()),minus_ema=float((v-old).mean()),
                               K8_full_admission=gate_lookup[(rule,s,route)]['K8_full_admission']))
    primary = [r for r in contrasts if r['contrast'] in ('revision_superiority','stable_noninferiority')]
    return dict(gates=gates,contrasts=contrasts,aggregate=aggregate_rows,permap=permap,
                primary=primary,joint_primary_success=all(r['admitted_inequality_pass'] for r in primary),
                draws=dict(seed=draw_seed,count=draws,history_count=n,stable_margin=margin,
                           fixed_parent_seeds=list(SEEDS),fixed_random_maps=5,
                           quantiles=[.0125,.9875],sampling_unit='whole history; jointly across methods/parents/maps'))
