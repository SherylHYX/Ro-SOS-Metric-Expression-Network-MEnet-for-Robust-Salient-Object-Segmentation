from pdb import set_trace
# set_trace()

from datasets import DUT, ECSSD, HKU_IS, MSRA1000, SOD

d_dict = {}

d_dict.update(DUT.data)
d_dict.update(ECSSD.data)
d_dict.update(HKU_IS.data)
d_dict.update(MSRA1000.data)
d_dict.update(SOD.data)

