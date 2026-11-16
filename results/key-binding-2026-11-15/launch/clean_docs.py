"""Polish only new session prose; leave historical blocks and user edits intact."""
import re
from pathlib import Path
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
def clean(text):
    for a,b in {'source5e8c618':'source `5e8c618`','executed5e8checkout':'executed `5e8c618` checkout',
                'onins093':'on ins093','onins086':'on ins086','maxlogit':'max logit','maxprob':'max probability',
                'fullscores':'full scores','pureweights':'pure weights','trainseen':'train-seen','devheldout':'dev-heldout',
                'permodel':'per model','bothkeys':'both keys','requestedkey':'requested key','observedvalue':'observed value',
                'allocatedGPUh':'allocated GPU hours','allocatedGPU':'allocated GPU','samebyte':'same byte',
                'originalpacket':'original packet','mainmetadata':'main metadata','sourceclone':'source clone',
                'allgroups':'all groups','allconditions':'all conditions','time-limit':'time limit',
                'nojobs':'no jobs','bothclosed':'both closed','Gitblobs':'Git blobs','Gitblobs':'Git blobs',
                'midpoint1000recovery-only':'midpoint 1000 recovery-only','Terminal2000only':'Terminal 2000 only'}.items():
        text=text.replace(a,b)
    text=re.sub(r'(?<=\d)(?=[KDN]\d)', ' ', text)
    text=re.sub(r'\b([KDN]\d+)(?=[a-z])',r'\1 ',text)
    protect=r'(`[^`]*`|PCG64|A6000|ins093|ins086|float32|uint8|int64|TF32|\b[KDN]\d+|\d+(?:\.\d+)?e[-+]\d+)'
    chunks=re.split(protect,text)
    for i in range(0,len(chunks),2):
        t=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',chunks[i])
        for a,b in {'%accuracy':'% accuracy','%in':'% in','bootstrap[':'bootstrap [',']intervals':'] intervals',
                    '%intervals':'% intervals','%CI':'% CI','0.84 M':'0.84M','underthisbudget':'under this budget',
                    'thisrecipe':'this recipe','newcommon':'new common','last100':'last 100','source50':'source 50',
                    'allsource':'all source','purestate':'pure state','inputbytes':'input bytes','full-value':'full-value'}.items():
            t=t.replace(a,b)
        chunks[i]=t
    return ''.join(chunks)
for name,boundary in [('docs/HANDOFF.md','**COMPLETE: sequence integration, acquisition FAILED, November 15.**'),
    ('docs/MEMBER_C_STATUS.md','## Latest: complete-sequence integration, failed acquisition')]:
    p=REPO/name;t=p.read_text(encoding='utf8');before,sep,after=t.partition(boundary);assert sep
    p.write_text(clean(before)+sep+after,encoding='utf8')
p=REPO/'README.md';t=p.read_text(encoding='utf8');start=t.index('The [key-binding encoding diagnostic]');end=t.index('\n\n',start)
p.write_text(t[:start]+clean(t[start:end])+t[end:],encoding='utf8')
p=Path(__file__).resolve().parent/'package_results.py'
t=p.read_text(encoding='utf8').replace('analysis/result_interpretation.md','analysis/independent_results_review.md')
p.write_text(t,encoding='utf8')
