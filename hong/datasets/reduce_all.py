import json
import numpy as np

from pdb import set_trace

names = ['DUT','ECSSD','HKU_IS','MSRA1000','SOD']
ext='json'
reduced_name = 'reduced'

dump = {}

dump['cols'] = ['max.abs', 'min.abs', 'median.abs', 'mean.abs', 'var.abs']

for name in names:
    with open('{}.{}'.format(name,ext), 'r') as f:
        stat = json.load(f)
    nstat = np.array(stat)
    # set_trace()
    dump[name] = nstat.mean(axis=0).tolist()

with open('{}.{}'.format(reduced_name,ext), 'w+') as f:
    json.dump(dump,f)

