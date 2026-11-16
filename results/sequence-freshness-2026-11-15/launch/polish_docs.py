"""Space numeric prose in this session's new documentation only."""
import re
from pathlib import Path
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
units=('registered','parameters','parameter','acquisition','attention','updates','update','answer','answers','forks','input','bytes','raw',
       'records','record','initializers','parents','trained','training','revision','replay','worlds','world','queries','query','scores','scorevectors','scores',
       'checkpoints','checkpoint','finite','terminal','optimizers','optimizer','source','files','file','gate','gates','models','model','steps','step',
       'histories','history','unique','scoped','selected','additional','both','vectors','signature','signatures','union','CPU','GPU','A6000','GiB','KiB',
       'percentile','percent','logical','neural','only','requires','words','on','plus','under','percase','per','required','retained','wide','readable','pairs',
       'folds','initial','main','mean','prompt','gradients','percent','test','tests','targets','short','literal','bounded','distractors','epochs','pp')
pattern=re.compile(r'(?<=\d)('+ '|'.join(sorted(set(units),key=len,reverse=True))+r')\b')
prefix=re.compile(r'\b(all|All|every|Every|each|Each|original|Original|unchanged|existing|Existing|and|at|of|on|with|after|under|than|versus|only|Only|through|across|fixed|FIFO|snapshot|step|Torch|seed|Job|job|source|pipeline|main|native|Native|wide|full|complete|hard|temporary|revised|against|CPU|GPU)(?=\d)')
def polish(text):
    parts=re.split(r'(`[^`]*`)',text)
    for i in range(0,len(parts),2):
        parts[i]=pattern.sub(r' \1',parts[i]);parts[i]=prefix.sub(r'\1 ',parts[i])
        parts[i]=parts[i].replace('%CI','% CI').replace('%CIs','% CIs').replace('pptrain','pp train').replace('8CPU','8 CPU')
    text=''.join(parts).replace('Source95da','Source `95da595`').replace('source95da595','source `95da595`').replace('LF95da','LF `95da595`')
    text=text.replace('PCG64draws','PCG64 draws').replace('WindowsCPU','Windows CPU').replace('LinuxGPU','Linux GPU').replace('GPUhours','GPU hours').replace('GPUh','GPU h').replace('onins080','on ins080').replace('nojobs','no jobs').replace('rawCSV','raw CSV').replace('Bonferroni2','the Bonferroni adjustment for two tests')
    text=re.sub(r'\b(K\d+)(?=[a-z])',r'\1 ',text)
    # Protect code, hashes and conventional alphanumeric model/metric identifiers.
    chunks=re.split(r'(`[^`]*`|PCG64|AdamW|A6000|ins080|[0-9a-f]{7,40}|\b[KDN]\d+|\d+(?:\.\d+)?e[-+]\d+)',text)
    for i in range(0,len(chunks),2):
        chunks[i]=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',chunks[i])
        chunks[i]=re.sub(r'(?<=[A-Za-z%)`]),(?=\d)',', ',chunks[i])
        chunks[i]=chunks[i].replace('×',' × ').replace('%both','% both').replace('%percentile','% percentile').replace('min(2,n)real','min(2,n) real')
    return ''.join(chunks)

for name in ('docs/SEQUENCE_FRESHNESS.md','docs/HANDOFF.md','docs/MEMBER_C_STATUS.md','README.md'):
    path=REPO/name;text=path.read_text(encoding='utf8')
    if name.endswith('HANDOFF.md'):
        before,marker,after=text.partition('**Bounded freshness comparison COMPLETE, November 15.**')
        assert marker;text=polish(before)+marker+after
    elif name.endswith('MEMBER_C_STATUS.md'):
        before,marker,after=text.partition('## Latest: bounded freshness repair')
        assert marker;text=polish(before)+marker+after
    elif name=='README.md':
        start=text.index('The [complete-sequence freshness pilot]');end=text.index('\n\n',start)
        text=text[:start]+polish(text[start:end])+text[end:]
    else:text=polish(text)
    path.write_text(text,encoding='utf8')
