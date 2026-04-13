
## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ineq': 27.512226104736328, 'feasibility_rate': 0.46875, 'mmse': 0.1214190348982811, 'smse': 4.993342876434326}
- Peak mean |A|: 1854.4


## Exp 2: Constraint-Quality Tradeoff
- mu0=0.001: {'ce_ineq': 0.5697542428970337, 'feasibility_rate': 0.46875, 'mmse': 0.06753410398960114, 'smse': 0.03642062470316887, 'time_per_sample_s': 15.695994771565893}
- mu0=0.01: {'ce_ineq': 1.3954193592071533, 'feasibility_rate': 0.015625, 'mmse': 0.04706590250134468, 'smse': 0.0457175150513649, 'time_per_sample_s': 15.587062500257161}
- mu0=0.1: {'ce_ineq': 13.028746604919434, 'feasibility_rate': 0.0, 'mmse': 0.03891092911362648, 'smse': 0.06941607594490051, 'time_per_sample_s': 15.300685534632066}
- mu0=1.0: {'ce_ineq': 4460929089536.0, 'feasibility_rate': 0.0, 'mmse': 3.55680844065835e+21, 'smse': 2.8594948308443368e+23, 'time_per_sample_s': 15.296222229357227}
- mu0=10.0: {'ce_ineq': inf, 'feasibility_rate': 0.0, 'mmse': inf, 'smse': inf, 'time_per_sample_s': 15.312816890451359}


## Exp 2: Constraint-Quality Tradeoff
- mu0=0.001: {'ce_ineq': 0.5697542428970337, 'feasibility_rate': 0.46875, 'mmse': 0.06753410398960114, 'smse': 0.03642062470316887, 'time_per_sample_s': 15.695994771565893}
- mu0=0.01: {'ce_ineq': 1.3954193592071533, 'feasibility_rate': 0.015625, 'mmse': 0.04706590250134468, 'smse': 0.0457175150513649, 'time_per_sample_s': 15.587062500257161}
- mu0=0.1: {'ce_ineq': 13.028746604919434, 'feasibility_rate': 0.0, 'mmse': 0.03891092911362648, 'smse': 0.06941607594490051, 'time_per_sample_s': 15.300685534632066}
- mu0=1.0: {'ce_ineq': 4460929089536.0, 'feasibility_rate': 0.0, 'mmse': 3.55680844065835e+21, 'smse': 2.8594948308443368e+23, 'time_per_sample_s': 15.296222229357227}
- mu0=10.0: {'ce_ineq': inf, 'feasibility_rate': 0.0, 'mmse': inf, 'smse': inf, 'time_per_sample_s': 15.312816890451359}
- eps=0.0001: {'ce_ineq': 2.8828024864196777, 'feasibility_rate': 0.421875, 'mmse': 0.05285624787211418, 'smse': 0.11880876868963242, 'time_per_sample_s': 15.561979785255971}
- eps=0.001: {'ce_ineq': 43.07789993286133, 'feasibility_rate': 0.265625, 'mmse': 0.24214023351669312, 'smse': 11.274916648864746, 'time_per_sample_s': 15.313725379732205}
- eps=0.01: {'ce_ineq': 1.3653781414031982, 'feasibility_rate': 0.0, 'mmse': 0.055310025811195374, 'smse': 0.047718267887830734, 'time_per_sample_s': 15.745363893176545}
- eps=0.1: {'ce_ineq': 0.8741371631622314, 'feasibility_rate': 0.0, 'mmse': 0.0481191985309124, 'smse': 0.03694818541407585, 'time_per_sample_s': 16.380647236743243}


## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ineq': 4.984040260314941, 'feasibility_rate': 0.359375, 'mmse': 0.05760209634900093, 'smse': 0.2758103907108307}
- Peak mean |A|: 1797.1


## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ineq': 0.5905802249908447, 'feasibility_rate': 0.25, 'mmse': 0.05330970510840416, 'smse': 0.04279816523194313}
- Peak mean |A|: 1598.4


## Exp 1: Main Comparison Table
- vanilla: {'ce_ineq': 0.1654396802186966, 'feasibility_rate': 0.0, 'mmse': 0.02336953580379486, 'smse': 0.047311730682849884, 'time_per_sample_s': 0.093003011555993}
- pcfm_equality: {'ce_ineq': 1.378955364227295, 'feasibility_rate': 0.0, 'mmse': 0.05184915289282799, 'smse': 0.040682584047317505, 'time_per_sample_s': 1.2727379009011202}
- ipcfm_a: {'ce_ineq': 0.25783824920654297, 'feasibility_rate': 0.0, 'mmse': 0.05997408553957939, 'smse': 0.03972342237830162, 'time_per_sample_s': 10.406233921297826}
- ipcfm_b: {'ce_ineq': 1.3644152879714966, 'feasibility_rate': 0.03125, 'mmse': 0.04286317899823189, 'smse': 0.04965754970908165, 'time_per_sample_s': 15.55104099548771}
- ipcfm_c: {'ce_ineq': 95.29696655273438, 'feasibility_rate': 0.140625, 'mmse': 0.9117085337638855, 'smse': 56.490882873535156, 'time_per_sample_s': 9.503946451601223}


## Exp 3: Runtime Breakdown
- vanilla: {'mean_ms_per_sample': 84.07717378577217, 'std_ms_per_sample': 2.0985539216660785, 'raw_ms': [87.04495211713947, 82.60385226458311, 82.58271697559394]}
- pcfm_equality: {'mean_ms_per_sample': 1285.9821564828355, 'std_ms_per_sample': 124.48269515578215, 'raw_ms': [1459.2824288120028, 1226.1494362610392, 1172.5146043754648]}
- ipcfm_a: {'mean_ms_per_sample': 10568.039026509117, 'std_ms_per_sample': 241.0021224122673, 'raw_ms': [10844.010562606854, 10256.840043148259, 10603.266473772237]}
- ipcfm_b: {'mean_ms_per_sample': 16925.85428651849, 'std_ms_per_sample': 974.9448623640131, 'raw_ms': [18304.42390337703, 16257.440165209118, 16215.698790969327]}
- ipcfm_c: {'mean_ms_per_sample': 9726.431337806085, 'std_ms_per_sample': 34.95983706692204, 'raw_ms': [9728.2604156062, 9682.729228661628, 9768.304369150428]}


## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ineq': 1.4560028314590454, 'feasibility_rate': 0.203125, 'mmse': 0.03775286301970482, 'smse': 0.05359356850385666}
- Peak mean |A|: 1729.3


## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ic': 0.0008436341886408627, 'ce_cl': 0.004548237193375826, 'ce_ineq': 0.862147331237793, 'feasibility_rate': 0.25, 'mmse': 0.031048327684402466, 'smse': 0.0454682931303978}
- Peak mean |A|: 1642.5


## Exp 1: Main Comparison Table
- vanilla: {'ce_ic': 3.067058563232422, 'ce_cl': 0.0557599775493145, 'ce_ineq': 0.15620818734169006, 'feasibility_rate': 0.0, 'mmse': 0.02863294817507267, 'smse': 0.059351347386837006, 'time_per_sample_s': 0.09235753350367304}
- pcfm_equality: {'ce_ic': 1.0818853297678288e-06, 'ce_cl': 0.0019783200696110725, 'ce_ineq': 1.8477692604064941, 'feasibility_rate': 0.0, 'mmse': 0.0509919673204422, 'smse': 0.050040993839502335, 'time_per_sample_s': 1.2621740462054731}
- ipcfm_a: {'ce_ic': 1.4707131867908174e-06, 'ce_cl': 0.0010822622571140528, 'ce_ineq': 0.43891286849975586, 'feasibility_rate': 0.0, 'mmse': 0.03283434733748436, 'smse': 0.03909998759627342, 'time_per_sample_s': 10.618181229030597}
- ipcfm_b: {'ce_ic': 0.0034460413735359907, 'ce_cl': 0.002826897194609046, 'ce_ineq': 1.4154356718063354, 'feasibility_rate': 0.0625, 'mmse': 0.03440763056278229, 'smse': 0.03747022524476051, 'time_per_sample_s': 15.971419423993211}
- ipcfm_c: {'ce_ic': nan, 'ce_cl': nan, 'ce_ineq': nan, 'feasibility_rate': 0.3125, 'mmse': nan, 'smse': nan, 'time_per_sample_s': 10.285486374166794}


## Exp 4: Active-Set Analysis
- n_steps=100, n_samples=64, eps=0.001
- Final metrics: {'ce_ic': 0.0003632267180364579, 'ce_cl': 0.0372513011097908, 'ce_ineq': 0.6971406936645508, 'feasibility_rate': 0.234375, 'mmse': 0.04989362880587578, 'smse': 0.03526550531387329}
- Peak mean |A|: 1608.2


## Exp 1: Main Comparison Table
- vanilla: {'ce_ic': 3.2399888038635254, 'ce_cl': 0.06826777011156082, 'ce_ineq': 0.18352201581001282, 'feasibility_rate': 0.0, 'mmse': 0.028183767572045326, 'smse': 0.05983695387840271, 'time_per_sample_s': 0.09275199114927091}
- pcfm_equality: {'ce_ic': 1.066740423993906e-06, 'ce_cl': 0.0018322458490729332, 'ce_ineq': 2.044430732727051, 'feasibility_rate': 0.0, 'mmse': 0.0452851727604866, 'smse': 0.05062635615468025, 'time_per_sample_s': 1.259723053430207}
- ipcfm_a: {'ce_ic': 1.2944749414600665e-06, 'ce_cl': 0.0009646032704040408, 'ce_ineq': 0.38423675298690796, 'feasibility_rate': 0.0, 'mmse': 0.036077529191970825, 'smse': 0.03381030634045601, 'time_per_sample_s': 10.672267773305066}
- ipcfm_b: {'ce_ic': 0.003121350659057498, 'ce_cl': 0.004854129161685705, 'ce_ineq': 1.1379249095916748, 'feasibility_rate': 0.015625, 'mmse': 0.03853471204638481, 'smse': 0.04358309879899025, 'time_per_sample_s': 16.362364682077896}
- ipcfm_c: {'ce_ic': 4.816585061144986e+16, 'ce_cl': nan, 'ce_ineq': inf, 'feasibility_rate': 0.15625, 'mmse': 5.546234438256852e+33, 'smse': inf, 'time_per_sample_s': 10.238857415344683}


## Exp 1: Main Comparison Table
- vanilla: {'ce_ic': 2.939548969268799, 'ce_cl': 0.0533822700381279, 'ce_ineq': 0.16271819174289703, 'feasibility_rate': 0.0, 'mmse': 0.021781297400593758, 'smse': 0.05055277794599533, 'time_per_sample_s': 0.170006632906734}
- pcfm_equality: {'ce_ic': 1.4469178495346569e-06, 'ce_cl': 0.00023417150077875704, 'ce_ineq': 0.49189817905426025, 'feasibility_rate': 0.0, 'mmse': 0.03241869434714317, 'smse': 0.00840726401656866, 'time_per_sample_s': 10.12943003399414}
- ipcfm_a: {'ce_ic': 2.281588194819051e-06, 'ce_cl': 7.293383532669395e-05, 'ce_ineq': 0.1262485831975937, 'feasibility_rate': 0.0, 'mmse': 0.022899990901350975, 'smse': 0.0060558319091796875, 'time_per_sample_s': 21.137351744924672}
- ipcfm_b: {'ce_ic': 0.0020308075472712517, 'ce_cl': 0.0013207225129008293, 'ce_ineq': 0.3319542407989502, 'feasibility_rate': 0.125, 'mmse': 0.025532592087984085, 'smse': 0.006260134745389223, 'time_per_sample_s': 32.906266176796635}
- ipcfm_c: {'ce_ic': 3.5522196292877197, 'ce_cl': 23958.365234375, 'ce_ineq': 147.19998168945312, 'feasibility_rate': 0.265625, 'mmse': 3.8789422512054443, 'smse': 87.43486022949219, 'time_per_sample_s': 21.76749221222417}


## Exp 1: Main Comparison Table
- vanilla: {'ce_ic': 3.427489757537842, 'ce_cl': 0.053505487740039825, 'ce_ineq': 0.15005452930927277, 'feasibility_rate': 0.0, 'mmse': 0.04010483995079994, 'smse': 0.06875848025083542, 'time_per_sample_s': 0.13270468076727554}
- pcfm_equality: {'ce_ic': 1.0875053249037592e-06, 'ce_cl': 0.0017696372233331203, 'ce_ineq': 2.0174450874328613, 'feasibility_rate': 0.0, 'mmse': 0.03525892272591591, 'smse': 0.05209236592054367, 'time_per_sample_s': 1.3491515388493152}
- ipcfm_a: {'error': 'Non-finite solution from solve'}
- ipcfm_b: {'ce_ic': 0.0017601761501282454, 'ce_cl': 0.0024745643604546785, 'ce_ineq': 0.9345360994338989, 'feasibility_rate': 0.05882353335618973, 'mmse': 0.022935867309570312, 'smse': 0.010073755867779255, 'time_per_sample_s': 30.477238890915817}
- ipcfm_c: {'error': 'Non-finite solution from solve'}

