import numpy as np
from scipy.stats import norm as ndist
from statsmodels.distributions import ECDF
from selection.covtest import covtest, reduced_covtest
from selection.affine import constraints, sample_from_constraints
from selection.discrete_family import discrete_family
import matplotlib.pyplot as plt

from constants import parameters, constraints
from full_data import power as power_full

def simulation(n, snr, pos, rho=0.25, ndraw=5000, burnin=1000):

    X, mu, beta = parameters(n, rho, pos)
    con, initial = constraints(X, pos)

    con.mean = snr * mu / np.sqrt(2)
    Z_selection = sample_from_constraints(con, initial, ndraw=ndraw, burnin=burnin)
    Z_inference_pos = np.random.standard_normal(Z_selection.shape[0]) + snr / np.sqrt(2)
    return (np.dot(X.T, Z_selection.T)[pos] + Z_inference_pos) / np.sqrt(2)

def power_onesided(n, snr, pos, rho=0.25, ndraw=10000,
                   muval = np.linspace(0,5,51), burnin=1000):

    S0 = simulation(n, 0, pos, rho=rho, ndraw=ndraw, burnin=burnin)
    W0 = np.ones(S0.shape)
    dfam0 = discrete_family(S0, W0)

    cutoff = dfam0.one_sided_acceptance(0, alternative='greater')[1]

    def UMPU_power_onesided(mu):
        return dfam0.ccdf(mu, cutoff)

    def sample_split_onesided(mu, alpha=0.05):
        cutoff = ndist.ppf(1 - alpha)
        if np.any(mu < 0):
            raise ValueError('mu is negative: in null hypothesis')
        power = 1 - ndist.cdf(cutoff - mu / np.sqrt(2))
        return np.squeeze(power)

    power_fig = plt.figure(figsize=(8,8))
    P_split = np.array(sample_split_onesided(muval))
    plt.plot(muval, P_split, label='Sample splitting', c='red', linewidth=5, alpha=0.5)
    power_ax = power_fig.gca()
    power_ax.set_ylabel('Power', fontsize=20)
    power_ax.legend(loc='lower right')
    power_ax.set_xlabel('Effect size $\mu$', fontsize=20)
    P_UMPU = np.array([UMPU_power_onesided(m) for m in muval])
    power_ax.plot(muval, P_UMPU, label=r'Selected using $i^*(Z_S)$', linewidth=5, alpha=0.5)
    P_full = power_full(n, snr, pos, rho=rho, muval=muval)[1]['full']
    power_ax.plot(muval, P_full, label=r'Selected using $i^*(Z)$', color='blue', linewidth=5, alpha=0.5)
    print UMPU_power_onesided(snr)
    power_ax.legend(loc='lower right')
    power_ax.set_xlim([0,5])
    power_ax.plot([snr,snr], [0,1], 'k--')
    return power_fig, {'umpu':P_UMPU, 'split':P_split}

def marginal(n, snr, pos, rho=0.25, ndraw=5000,
             burnin=1000, nsim=5000, sigma=1.):

    X, mu, beta = parameters(n, rho, pos)

    Psplit = []
    Pselect = []
    hypotheses = []


    for _ in range(nsim):
        Y_select = (snr * mu / np.sqrt(2) + np.random.standard_normal(n)) * sigma
        con, _, select_pos, sign = covtest(X, Y_select, sigma=sigma, exact=True)

        cond_ncp = snr * np.dot(X.T[select_pos], mu) / np.sqrt(2) * sign

        correct = (sign == +1) and (pos == select_pos)
        hypotheses.append(correct)
        Y_null = sample_from_constraints(con, Y_select, ndraw=ndraw, burnin=burnin)
        Z_null = (np.dot(X.T[select_pos], Y_null.T) + sigma * np.random.standard_normal(ndraw)) / np.sqrt(2)
        Z_inference = sigma * (cond_ncp + np.random.standard_normal())
        Z_observed = (np.dot(X.T[select_pos], Y_select) * sign + Z_inference) / np.sqrt(2)
        dfam = discrete_family(Z_null, np.ones(Z_null.shape))
        Pselect.append(dfam.ccdf(0, Z_observed))
        if sign == +1:
            Psplit.append(ndist.sf(Z_inference / sigma))
        else:
            Psplit.append(ndist.cdf(Z_inference / sigma))

    Ugrid = np.linspace(0,1,101)

    Psplit = np.array(Psplit)
    Pselect = np.array(Pselect)
    hypotheses = np.array(hypotheses, np.bool)

    # plot of marginal distribution of p-values

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.gca()
    ax1.plot(Ugrid, ECDF(Psplit)(Ugrid), label='Sample splitting', c='red', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pselect)(Ugrid), label='Selected using $i^*(Z_S)$', c='blue', linewidth=5, alpha=0.5)
    ax1.set_xlabel('P-value, $p$', fontsize=20)
    ax1.set_ylabel('ECDF($p$)', fontsize=20)
    ax1.plot([0.05,0.05],[0,1], 'k--')
    ax1.legend(loc='lower right')
    
    # conditional distribution of p-values
    # conditioned on selection choosing correct position and sign

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.gca()
    ax2.plot(Ugrid, ECDF(Psplit[hypotheses])(Ugrid), label='Sample splitting', c='red', linewidth=5, alpha=0.5)
    ax2.plot(Ugrid, ECDF(Pselect[hypotheses])(Ugrid), label='Selected using $i^*(Z_S)$', c='blue', linewidth=5, alpha=0.5)
    ax2.set_xlabel('P-value, $p$', fontsize=20)
    ax2.set_ylabel('ECDF($p$)', fontsize=20)
    ax2.plot([0.05,0.05],[0,1], 'k--')
    ax2.legend(loc='lower right')

    dbn1 = {}
    dbn1['split'] = Psplit
    dbn1['select'] = Pselect
    dbn1['hypotheses'] = hypotheses

    return fig1, fig2, dbn1

# # ## Selection intervals
# # 
# # To create the selection intervals, it helps to have data sampled near our observation.
# # We will store these, then form two intervals one based on data to the left, the other
# # on data to the right. In this simple example, we just average the endpoints based on where
# # the observation falls into the interval. A better way would be to pool the sufficient statistics
# # and use importance weights.

# # In[ ]:

# S = {}
# dfam = {}
# for i in range(7):
#     S[i] = sample(i, ndraw=100000)
#     W = np.ones(S[i].shape)
#     dfam[i] = discrete_family(S[i], W)
#     dfam[i].theta = 0


# # In[ ]:

# [dfam0.interval(3, randomize=False) for _ in range(4)]


# # In[ ]:

# Zvals = np.linspace(0,6,101)
# UMPU_intervals = []
# twotailed_intervals = []
# for z in Zvals:
#     z1, z2 = np.floor(z), np.ceil(z)
#     # weight for convex combination
#     w = (z - z1)
    
#     u1, l1 = dfam[z1].interval(z, randomize=True, auxVar=0.5)
#     u2, l2 = dfam[z2].interval(z, randomize=True, auxVar=0.5)
#     u1, l1 = u1 + z1, l1 + z1
#     u2, l2 = u2 + z2, l2 + z2
#     u, l = ((1-w)*u1+w*u2), ((1-w)*l1+w*l2)
#     UMPU_intervals.append((u,l))
    
#     u1, l1 = dfam[z1].equal_tailed_interval(z, auxVar=0.5)
#     u2, l2 = dfam[z2].equal_tailed_interval(z, auxVar=0.5)
#     u1, l1 = u1 + z1, l1 + z1
#     u2, l2 = u2 + z2, l2 + z2
#     u, l = ((1-w)*u1+w*u2), ((1-w)*l1+w*l2)
#     twotailed_intervals.append((u,l))


# # In[ ]:

# from scipy.stats import norm as ndist

# UMPU_intervals = np.array(UMPU_intervals)
# twotailed_intervals = np.array(twotailed_intervals)
# interval_fig = plt.figure(figsize=(8,8))
# interval_ax = interval_fig.gca()
# interval_ax.plot(Zvals, UMPU_intervals[:,0], c="green", linewidth=3, label='Reduced model UMAU')
# interval_ax.plot(Zvals, UMPU_intervals[:,1], c='green', linewidth=3)
# interval_ax.plot(Zvals, twotailed_intervals[:,0], c="purple", linewidth=3, label='Reduced model equal-tailed')
# interval_ax.plot(Zvals, twotailed_intervals[:,1], c='purple', linewidth=3)

# interval_ax.plot(Zvals, Zvals - ndist.ppf(0.975) * np.sqrt(2), c='red', linewidth=3, label='Sample splitting')
# interval_ax.plot(Zvals, Zvals + ndist.ppf(0.975) * np.sqrt(2), c='red', linewidth=3)
# interval_ax.plot(Zvals, Zvals - ndist.ppf(0.975), '--', c='blue', linewidth=1, label='Nominal')
# interval_ax.plot(Zvals, Zvals + ndist.ppf(0.975), '--', c='blue', linewidth=1)
# interval_ax.set_xlabel(r'Observed statistic: $T_{i^*}$ for reduced, $\sqrt{2} \cdot T_{i^*,2}$ for splitting)', fontsize=20)
# interval_ax.legend(loc='upper left')
# interval_ax.plot([np.sqrt(2*np.log(n))]*2,[-6,10], 'k--')
# interval_fig.savefig('sample_splitting_intervals.pdf')



def main():

    fig1, fig2, dbn = marginal(20, 3., 3, nsim=1000)
    full = np.load('pval_20.npz')
    Ugrid = np.linspace(0,1,101)

    ax1 = fig1.gca()
    ax1.plot(Ugrid, ECDF(full['known'])(Ugrid), label=r'Selected using $i^*(Z)$', c='green', linewidth=5, alpha=0.5)
    ax1.legend(loc='lower right')

    ax2 = fig2.gca()
    ax2.plot(Ugrid, ECDF(full['known'][full['hypotheses']])(Ugrid), label=r'Selected using $i^*(Z)$', c='green', linewidth=5, alpha=0.5)
    ax2.legend(loc='lower right')

    fig1.savefig('splitting_marginal_1sparse.pdf')
    fig2.savefig('splitting_conditional_1sparse.pdf')

    #power_one = power_onesided(20, 3., 3, ndraw=4000000, burnin=100000)[0]
    #power_one.savefig('splitting_onesided_power.pdf')

if __name__ == '__main__':
    main()

# # ## Data for illustrative purposes
# # 
# # At an effect size of 3, sample splitting has power roughly 60% of rejecting (conditonal on the first position)
# # being largest. Let's sample some data from this distribution to use as illustration.

# # In[ ]:

# np.random.seed(10)
# con.mean = np.zeros(n+1)
# snr = 3
# con.mean = snr * mu_vec / np.sqrt(2)
# data_selection = sample_from_constraints(con, initial, ndraw=80000, burnin=5000)[-1]
# data_inference = np.dot(X.T, np.random.standard_normal(n)) + snr * np.dot(X.T, mu_vec) / np.sqrt(2)
# data_fig = plt.figure(figsize=(8,8))
# data_ax = data_fig.gca()
# data_ax.scatter(np.arange(n), data_inference, c='b', marker='o', s=100)
# data_ax.scatter(np.arange(n), data_selection, c='r', marker='+', s=100)
# data_ax.plot(np.arange(n), data_inference, c='b', label=r'Inference: $Z_I$', alpha=0.5, linewidth=3)
# data_ax.plot(np.arange(n), data_selection, c='r', label=r'Selection: $Z_S$', alpha=0.5, linewidth=3)
# data_ax.set_xlim([-0.5,20.5])
# data_ax.legend(fontsize=20)
# data_ax.set_xticks([4,9,14,19])
# data_ax.set_xticklabels([5,10,15,20], fontsize=20)
# data_ax.set_xlabel('Index', fontsize=20)
# data_ax.set_ylabel('Observed data', fontsize=20)
# data_ax.plot([3,3],[-3,5], 'k--')
# data_ax.set_ylim([-3,4])
# data_fig.savefig('data_instance.pdf')


# # In[ ]:

# get_ipython().magic(u'load_ext rmagic')
# get_ipython().magic(u'R -i X,data_selection,data_inference')


# # In[ ]:

# print np.linalg.norm(X[:,3])


# # In[ ]:

# get_ipython().run_cell_magic(u'R', u'', u'Z = (data_selection + data_inference) / sqrt(2)\nclassical_model = lm(Z ~ X[,1] + X[,2] + X[,3] + X[,4] + X[,5] + X[,6] + X[,7] - 1)\nanova(classical_model)')


# # In[ ]:

# get_ipython().run_cell_magic(u'R', u'', u"selection_model = lm(data_selection ~ X[,1] + X[,2] + X[,3] + X[,4] + X[,5] + X[,6] + X[,7] + X[,8] + X[,9] +\n                 X[,10] + X[,11] + X[,12] + X[,13] + X[,14] + X[,15] + X[,16] + X[,17] + X[,18] + X[,19] + X[,20] - 1)\nstep(lm(data_selection ~ -1), list(upper=~ X[,1] + X[,2] + X[,3] + X[,4] + X[,5] + X[,6] + X[,7] + X[,8] + X[,9] +\n                 X[,10] + X[,11] + X[,12] + X[,13] + X[,14] + X[,15] + X[,16] + X[,17] + X[,18] + X[,19] + X[,20] - 1),\n     steps=1, direction='forward')\n\ninference_model = glm(data_inference ~ X[,4] - 1)\nprint(vcov(inference_model))\nprint(summary(inference_model))\neffect = sum(X[,4]*data_inference)\nlower = effect - qnorm(0.975)\nupper = effect + qnorm(0.975)\nprint(data.frame(effect, lower, upper))")


# # ## Non-adaptive inference

# # In[ ]:

# Zmax = np.max(np.fabs(np.dot(X.T, np.random.standard_normal((X.shape[0], 10000)))), 0)
# nonadapt_fig = plt.figure(figsize=(8,8))
# nonadapt_ax = nonadapt_fig.gca()
# Zgrid = np.linspace(0,5,201)
# from scipy.stats import norm as ndist
# pmax = 2 * ndist.sf(Zmax)
# nonadapt_ax.plot(grid, ECDF(pmax)(grid), linewidth=3, alpha=0.5)
# nonadapt_ax.set_xlabel('P-value, $p$', fontsize=20)
# nonadapt_ax.set_ylabel('ECDF($p$)', fontsize=20)
# nonadapt_fig.savefig('nonadapt_pvalue.pdf')


