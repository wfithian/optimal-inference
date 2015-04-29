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
    return sample[:nsample]

def interval(mu, ndraw=100000, keep_every=100):

    if not os.path.exists('lengths%0.2f.npz' % mu):
        lengths = []
    else:
        lengths = list(np.load('lengths%0.2f.npz' % mu)['lengths'])

    big_sample = draw_sample(mu, cutoff, nsample=50000)[:50000]
    mean, scale = big_sample.mean(), big_sample.std()
    big_sample -= mean
    big_sample /= scale

    dbn = discrete_family(big_sample, np.ones_like(big_sample))
    dbn.theta = 0.
    new_sample = draw_sample(mu, cutoff, nsample=2500)[:2500]
    for i, s in enumerate(new_sample):
        try:
            _interval = dbn.equal_tailed_interval((s - mean) / scale)
            lengths.append(np.fabs(_interval[1] - _interval[0]) / scale)
        except:
            print 'exception raised'
        if i % 20 == 0 and i > 0:
            print np.median(lengths), np.mean(lengths)
            np.savez('lengths%0.2f' % mu, **{'lengths':lengths,'mu':mu})
        if i % 1000 == 0 and i > 0:
            big_sample = draw_sample(mu, cutoff, nsample=50000)[:50000]
            mean, scale = big_sample.mean(), big_sample.std()
            big_sample -= mean
            big_sample /= scale
        dbn.theta = 0.
        print i
    return (np.mean(lengths), np.std(lengths), np.median(lengths))

def main():
    muvals = np.linspace(-2, 9, 23)[::-1]
    L = []
    np.random.shuffle(muvals)
    for mu in muvals:
        print 'trying %0.2f' % mu
        for f in glob('lengths*npz'):
            d = np.load(f)
            if d['mu'] == mu and d['lengths'].shape[0] > 50000:
                print '%0.2f already done' % mu
            else:
                interval(mu)

def plot():

    results = []
    for f in glob('lengths*npz'):
        d = np.load(f)
        l = d['lengths']
        l = l[l>0.]
        print d['mu'], l.shape
        results.append([d['mu'], l.mean()])

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
    f.savefig('figure_b.pdf')
    output = np.array(zip(muvals, mean_length))
    np.savetxt('equal_tailed_lengths.csv', output, delimiter=',')
    
if __name__ == '__main__':
    # main()
    pass
