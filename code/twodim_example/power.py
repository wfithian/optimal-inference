import os
import numpy as np
import matplotlib.pyplot as plt
from selection import affine 
from selection.discrete_family import discrete_family
from scipy.stats import norm as ndist

cutoff = ndist.ppf(0.95)

null_constraint = affine.constraints(np.array([[-1,0.]]), np.array([-cutoff]))
null_sample = affine.sample_from_constraints(null_constraint, np.array([4,2.]),
                                             ndraw=100000).sum(1)
null_dbn = discrete_family(null_sample, np.ones_like(null_sample))

def power(mu, ndraw=100000, keep_every=100):
    constraint = affine.constraints(np.array([[-1,0.]]), np.array([-cutoff]))
    constraint.mean = np.array([mu,mu])
    sample = affine.sample_from_constraints(constraint, np.array([4,2.]),
                                            ndraw=ndraw)[::keep_every]
    print sample.mean(0)
    sample = sample.sum(1)
    decisions = []
    for s in sample:
        decisions.append(null_dbn.one_sided_test(0, s, alternative='greater'))
    print np.mean(decisions)
    return np.mean(decisions)

if not os.path.exists('power_curve.npy'):
    muvals = np.linspace(0, 5, 21)
    P = [power(mu, ndraw=100000, keep_every=25) for mu in muvals]
    np.save('power_curve.npy', np.vstack([muvals, P]))
else:
    muvals, P = np.load('power_curve.npy')

plt.clf()
plt.plot(muvals, P, 'k', linewidth=2, label='Selective $z$ test')
plt.plot(muvals, [ndist.sf(ndist.ppf(0.95) - mu) for mu in muvals],
         c='red', label='Sample splitting', linewidth=2)
ax = plt.gca()
ax.set_xlabel(r'$\mu$', fontsize=20)
ax.set_ylabel(r'Power($\mu$)', fontsize=20)
ax.legend(loc='lower right')
f = plt.gcf()
f.savefig('figure_c.pdf')
