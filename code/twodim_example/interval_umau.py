import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from selection import affine 
from selection.discrete_family import discrete_family
from scipy.stats import norm as ndist
from sklearn.isotonic import IsotonicRegression

cutoff = 3.
null_constraint = affine.constraints(np.array([[-1,0.]]), np.array([-cutoff]))
null_sample = affine.sample_from_constraints(null_constraint, np.array([4,2.]),
                                             ndraw=100000).sum(1)
null_dbn = discrete_family(null_sample, np.ones_like(null_sample))

def draw_sample(mu, cutoff, nsample=10000):
    if mu >= cutoff - 4:
        sample = []
        while True:
            candidate = np.random.standard_normal(1000000) + mu
            candidate = candidate[candidate > cutoff]
            sample.extend(candidate)
            if len(sample) > nsample:
                break
        sample = np.array(sample)
        sample += np.random.standard_normal(sample.shape) + mu
    else:
        constraint = affine.constraints(np.array([[-1,0.]]), np.array([-cutoff]))
	constraint.mean = np.array([mu,mu])
        sample = affine.sample_from_constraints(constraint, np.array([cutoff + 0.1,0]),
                                                ndraw=2000000,
                                                direction_of_interest=np.array([1,1.]))
        sample = sample.sum(1)[::(2000000/nsample)]
    return sample

def form_samples(nsample=10000):
    samples = {}
    for mu in range(-6, 13):
        label = 'mu%d' % mu
        print label
        samples[label] = draw_sample(mu, cutoff, nsample=nsample)[:nsample]
    return samples

def form_dbn(mu, samples):
    pts = np.arange(-6,13)
    keep = np.fabs(pts - mu) <= 2
    pts = pts[keep]
    _samples = np.hstack([samples['mu%d' % l] for l in pts])
    _log_weights = np.hstack([(mu-l)*samples['mu%d' % l] for l in pts])
    _weights = np.exp(_log_weights)
    dbn = discrete_family(_samples, _weights)
    dbn.theta = 0.
    return dbn

def interval(mu, ndraw=100000, keep_every=100):
    #dbn = form_dbn(mu, samples)

    if not os.path.exists('umau_lengths%0.2f.npz' % mu):
        lengths = []
    else:
        lengths = list(np.load('umau_lengths%0.2f.npz' % mu)['lengths'])

    if mu < 10:
        big_sample = draw_sample(mu, cutoff, nsample=50000)[:50000]
        mean, scale = big_sample.mean(), big_sample.std()
        big_sample -= mean
        big_sample /= scale

        dbn = discrete_family(big_sample, np.ones_like(big_sample))
        dbn.theta = 0.
        new_sample = draw_sample(mu, cutoff, nsample=2500)[:2500]
        for i, s in enumerate(new_sample):
            try:
                _interval = dbn.interval((s - mean) / scale)
                lengths.append(np.fabs(_interval[1] - _interval[0]) / scale)
            except:
                print 'exception raised'
            if i % 20 == 0 and i > 0:
                print np.median(lengths), np.mean(lengths)
                np.savez('umau_lengths%0.2f' % mu, **{'lengths':lengths,'mu':mu})
            if i % 1000 == 0 and i > 0:
                big_sample = draw_sample(mu, cutoff, nsample=50000)[:50000]
                mean, scale = big_sample.mean(), big_sample.std()
                big_sample -= mean
                big_sample /= scale
            print i
    else:
        for i in range(2500):
            big_sample = draw_sample(mu, cutoff, nsample=50000)[:50000]
            s = big_sample[-1]
            big_sample = big_sample[:-1]
            mean, scale = big_sample.mean(), big_sample.std()
            big_sample -= mean
            big_sample /= scale
            s = (s - mean) / scale
            dbn = discrete_family(big_sample, np.ones_like(big_sample))
            try:
                _interval = dbn.interval(s)
                lengths.append(np.fabs(_interval[1] - _interval[0]) / scale)
            except:
                print 'exception raised'
            print i
            if i % 10 == 0 and i > 0:
                print np.median(lengths), np.mean(lengths)
                np.savez('umau_lengths%0.2f' % mu, **{'lengths':lengths,'mu':mu})

    print 'final', np.mean(lengths)
    return (np.mean(lengths), np.std(lengths), np.median(lengths))

if not os.path.exists('interval_samples.npz'):
    samples = form_samples()
    np.savez('interval_samples.npz', **samples)
else:
    samples = np.load('interval_samples.npz')

def main():
    muvals = np.linspace(-2, 9, 23)[::-1]
    L = []
    np.random.shuffle(muvals)
    for mu in muvals:
        print 'trying %0.2f' % mu
        for f in glob('umau_lengths*npz'):
            d = np.load(f)
            if d['mu'] == mu and d['lengths'].shape[0] > 5000:
                print '%0.2f already done' % mu
            else:
                interval(mu)

def plot():

    results = []
    for f in glob('umau_lengths*npz'):
        d = np.load(f)
        l = d['lengths']
        l = l[~np.isnan(l)]
        l = l[np.isfinite(l)]
        l = l[l>0]
        results.append([d['mu'], l.mean()])
    for f in glob('miller/lengths*npz'):
        d = np.load(f)
        if d['mu'] not in [r[0] for r in results]:
            l = d['lengths']
            l = l[np.isfinite(l)]
            l = l[~np.isnan(l)]
            l = l[l>0]
            results.append([d['mu'], l.mean()])
        else:
            idx = [r[0] for r in results].index(d['mu'])
            l = d['lengths']
            l = l[np.isfinite(l)]
            l = l[~np.isnan(l)]
            l = l[l>0]
            results[idx][1] = 0.5 * (results[idx][1] + l.mean())
    results = sorted(results)
    results = np.array(results).T
    muvals, mean_length = results
    f = plt.figure()
    f.clf()
    ax = f.gca()
    iso = IsotonicRegression(increasing=False)
    mean_length_iso = iso.fit_transform(np.arange(mean_length.shape[0]), mean_length)    
    ax.plot(muvals, mean_length, 'k', linewidth=2, label='UMAU')
    ax.plot([muvals.min(), muvals.max()], [2*ndist.ppf(0.975)]*2, c='red', label='Sample splitting', linewidth=2)
    ax.plot([muvals.min(), muvals.max()], [np.sqrt(2)*ndist.ppf(0.975)]*2, 'k--')
    ax.set_xlabel(r'$\mu$', fontsize=20)
    ax.set_ylabel(r'E(|CI($\mu$)|)', fontsize=20)
    ax.legend(loc='lower right')
    ax.set_ylim([0,4])
    ax.set_xlim([-2,9])
    f.savefig('figure_b_umau.pdf')

if __name__ == '__main__':
    # main()
    pass
