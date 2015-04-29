import numpy as np, csv, hashlib, os.path
import pandas as pd
import uuid
from selection.algorithms.tests.test_lasso import test_data_carving

from data_carving import split_vals, vals, df, dname

# run on commit with SHA b91c434e74ad9d623d256db9f20e66c643504239 on jonathan-taylor/selective-inference

np.random.seed(0)

# how many points do we want for each fraction

min_sample_size = 100000

vals = vals + [('lam_frac', 2.),
               ('split_frac', 0.9)]

dtype = np.dtype([('n', np.int), 
                  ('p', np.int), 
                  ('s', np.int), 
                  ('sigma', np.int), 
                  ('rho', np.float), 
                  ('snr', np.float), 
                  ('lam_frac', np.float), 
                  ('split_frac', np.float), 
                  ('method', 'S5'), 
                  ('null', np.bool), 
                  ('pval', np.float),
                  ('uuid', 'S40')])

num_except = 0

for i in range(5000):
    for split_frac in split_vals[::-1]:
        opts = dict(vals)
        opts['split_frac'] = split_frac
        identifier = str(uuid.uuid1())
        fname = '%s/results_split_%0.2f.npy' % (dname, split_frac)
        opts['df'] = df # degrees of freedom for noise
        opts['compute_intervals'] = False
        opts['ndraw'] =  8000
        opts['burnin'] = 2000

        test = not os.path.exists(fname)
        if not test:
            prev_results = np.load(fname)
            if split_frac not in [0.3, 0.4, 0.5]:
                test = prev_results.shape[0] < min_sample_size
            elif split_frac in [0.3, 0.4]:
                test = prev_results.shape[0] < 5000
            else:
                test = prev_results.shape[0] < 10000
        if test:
            try:
                results = test_data_carving(**opts)
                (null_carve, 
                 null_split, 
                 alt_carve,
                 alt_split,
                 counter) = results[-1][:-4]

                FP_cur = [result[-1] for result in results]
                TP_cur = [result[-2] for result in results]

                print FP_cur, TP_cur
                params = [v for _, v in vals]
                params[-1] = split_frac
                results = []

                if os.path.exists("%s/discovery_rates.npy" % dname):
                    prev_results = np.load('%s/discovery_rates.npy' % dname)
                    disc = np.empty(prev_results.shape[0] + len(FP_cur), 
                                    prev_results.dtype)
                    disc[:-len(FP_cur)] = prev_results
                    disc[-len(FP_cur):]['split_frac'] = split_frac
                    disc[-len(FP_cur):]['FP'] = FP_cur
                    disc[-len(FP_cur):]['TP'] = TP_cur
                    np.save('%s/discovery_rates.npy' % dname, disc)

                else:
                    dtype_disc = np.dtype([('split_frac', np.float),
                                           ('FP', np.int),
                                           ('TP', np.int)])
                    disc = np.empty(len(FP_cur), dtype_disc)
                    disc['FP'] = FP_cur
                    disc['TP'] = TP_cur
                    disc['split_frac'][:] = split_frac
                    np.save('%s/discovery_rates.npy' % dname, disc)

                if os.path.exists('%s/screening.npy' % dname):
                    prev_results = np.load('%s/screening.npy' % dname)
                    screening = np.empty(prev_results.shape[0]+1, 
                                          prev_results.dtype)
                    screening[:-1] = prev_results
                    screening[-1] = (split_frac, counter, identifier)
                    np.save('%s/screening.npy' % dname, screening)
                else:
                    dtype_screen = np.dtype([('split', np.float),
                                             ('counter', np.float),
                                             ('uuid', 'S40')])
                    screening = np.array([(split_frac, counter, identifier)],
                                         dtype_screen)
                    np.save('%s/screening.npy' % dname, screening)

                results.extend([tuple(params) + ('carve', True, p, identifier) 
                                for p in null_carve])
                results.extend([tuple(params) + ('split', True, p, identifier)
                                for p in null_split])
                results.extend([tuple(params) + ('carve', False, p, identifier) 
                                for p in alt_carve])
                results.extend([tuple(params) + ('split', False, p, identifier) 
                                for p in alt_split])

                rec_results = np.array(results, dtype)
                if os.path.exists(fname):
                    prev_results = np.load(fname)
                    rec_results = np.hstack([prev_results, rec_results])
                np.save(fname, rec_results)
                print rec_results.shape, 1. / screening[screening['split'] == split_frac]['counter'].mean(), fname
            except:
                num_except += 1
                print("exception raised: %d" % num_except)
                pass
    print "num exception: %d" % num_except
