import utils
import numpy as np

mf_name = {1: 'results/2015111522493294/fold_%d_mf.pkl',
           0.6: './results/2016020312181037/fold_%d_mf.pkl',
           0.8: './results/2016020222205294/fold_%d_mf.pkl'}


for coverage in [1,0.8,0.6]:
    factors = []
    print(coverage)
    for fold in range(0,5):
        try:
            P,Q = utils.read_gzpickle(mf_name[coverage] % fold)
            print ('factors: ', P.shape[1])
            factors += [P.shape[1]]
        except KeyError:
            pass
    print(np.mean(factors), np.std(factors))
    print(np.mean(factors)/943)
    print(np.mean(factors)/1682)

        
