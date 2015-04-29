import numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.mlab import rec2csv

split_vals = ([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.88, 0.9] + 
              [0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                       0.97, 0.98, 0.99, 1.00, 0.87, 0.88]*4)
vals = [('n', 100),
        ('p', 200),
        ('s', 7),
        ('sigma', 5.),
        ('rho', 0.3),
        ('snr', 7.)
        ]
opts = dict(vals)

df = 5
if df < np.inf:
    dname = 'df_%d' % df
else:
    dname = 'gaussian'

def summary(df, save=True):

    if df < np.inf:
        dname = 'df_%d' % df
    else:
        dname = 'gaussian'

    # put in point at 0.3

    (split_frac, level_carve, level_split, 
     power_carve, power_split, p_screen) = \
        (0.3, 0.05, 0.05, 1., 1., 0.)
    results = [(split_frac, level_carve, level_split, power_carve, 
                power_split, p_screen, -1, -1, -1, -1, -1, -1)]
    
    if True: # try:
        screen = np.load('%s/screening.npy' % dname)
        disc = np.load('%s/discovery_rates.npy' % dname)

        for split_frac in sorted(np.unique(split_vals)):
            fname = '%s/results_split_%0.2f.npy' % (dname, split_frac)
            data = np.load(fname)
            null_carve = np.array([d['pval'] for d in data if d['method'] == 'carve' 
                                   and d['null'] == True])
            null_split = np.array([d['pval'] for d in data if d['method'] == 'split' 
                                   and d['null'] == True])
            alt_carve = np.array([d['pval'] for d in data if d['method'] == 'carve' 
                                   and d['null'] == False])
            alt_split = np.array([d['pval'] for d in data if d['method'] == 'split' 
                                   and d['null'] == False])

            disc_rate = disc[disc['split_frac'] == split_frac]
            FP = np.nanmean(disc_rate['FP']) # V
            TP = np.nanmean(disc_rate['TP']) # R-V

            R = FP + TP
            FDP = np.nanmean(FP / np.maximum(R, 1))

#             for _, trial in df.groupby('uuid'):
#                 FP_carve.append(((trial['method'] == 'carve') * (trial['null'] == True)).sum())
#                 FP_split.append(((trial['method'] == 'split') * (trial['null'] == True)).sum())
#                 TP_carve.append(((trial['method'] == 'carve') * (trial['null'] == False)).sum())
#                 TP_split.append(((trial['method'] == 'split') * (trial['null'] == False)).sum())

#             FP_carve = np.array(FP_carve)
#             FP_split = np.array(FP_split)
#             TP_carve = np.array(TP_carve)
#             TP_split = np.array(TP_split)

#             FDP_carve.append(FP_carve * 1. / (FP_carve + TP_carve))
#             FDP_split.append(FP_split * 1. / (FP_split + TP_split))

            power_carve = np.nanmean(alt_carve < 0.05)
            power_split = np.nanmean(alt_split < 0.05)
            level_carve = np.nanmean(null_carve < 0.05)
            level_split = np.nanmean(null_split < 0.05)

            p_screen = 1. / np.nanmean(screen[screen['split'] == split_frac]['counter'])
            result = (split_frac, 
                      level_carve, 
                      level_split, 
                      power_carve, 
                      power_split, 
                      p_screen, 
                      (data['null'] == True).sum(), 
                      (data['null'] == False).sum(), 
                      len(set(data['uuid'])),
                      FP,
                      TP,
                      FDP,
                      )
            results.append(result)
            print split_frac

        R = np.array(results, np.dtype([ \
                    ('split', np.float),
                    ('level_carve', np.float),
                    ('level_split', np.float),
                    ('power_carve', np.float),
                    ('power_split', np.float),
                    ('p_screen', np.float),
                    ('count_null', np.int),
                    ('count_alt', np.int),
                    ('ntrial', np.int),
                    ('fp', np.float),
                    ('tp', np.float),
                    ('fdp', np.float)
                    ]))

        if save:
            np.save('%s/summary.npy' % dname, R)
            rec2csv(R, '%s/summary.csv' % dname)
            os.system('cd %s; R CMD BATCH makeRplots.r' % dname)

        plt.clf()
        plt.plot(R['split'], R['p_screen'])
        plt.plot(R['split'], R['power_split'])
        plt.plot(R['split'], R['power_carve'])
        plt.plot(R['split'], R['level_split'])
        plt.plot(R['split'], R['level_carve'])
        plt.plot([0.3,1],[0.05,0.05], 'k--')
        plt.savefig('%s/summary.pdf' % dname)

    else: #except:
        print 'no results yet'
        pass
