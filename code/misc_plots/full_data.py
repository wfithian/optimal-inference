import numpy as np
from scipy.stats import norm as ndist
from statsmodels.distributions import ECDF
from selection.covtest import covtest, reduced_covtest
from selection.affine import constraints, sample_from_constraints
from selection.discrete_family import discrete_family
import matplotlib.pyplot as plt

from constants import parameters, constraints

def simulation(n, snr, pos, rho=0.25, nsim=5000, sigma=1.5):

    # Design, mean vector and parameter vector

    X, mu, beta = parameters(n, rho, pos)

    Pcov = []
    Pexact = []
    Pu = []
    Pr = []
    Pfixed = []
    Pmax = []
    hypotheses = []
    
    
    # Set seed

    np.random.seed(0)

    # Max test

    max_stat = np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0) * sigma
    max_fam = discrete_family(max_stat, np.ones(max_stat.shape))
    max_fam.theta = 0

    for i in range(nsim):
        Y = (snr * mu + np.random.standard_normal(n)) * sigma
        Z = np.dot(X.T, Y)

        # did this find the correct position and sign?
        correct = np.all(np.less_equal(np.fabs(Z), Z[pos]))
        hypotheses.append(correct)

        Pcov.append(covtest(X, Y, sigma=sigma, exact=False)[1])
        Pexact.append(covtest(X, Y, sigma=sigma, exact=True)[1])
        Pfixed.append(2 * ndist.sf(np.fabs(np.dot(X.T, Y))[pos] / sigma))
        Pu.append(reduced_covtest(X, Y, burnin=500, ndraw=5000)[1])
        Pr.append(reduced_covtest(X, Y, burnin=500, ndraw=5000, sigma=sigma)[1])
        p = max_fam.ccdf(0, np.fabs(np.dot(X.T, Y)).max())
        Pmax.append(p)

    Ugrid = np.linspace(0,1,101)

    Pcov = np.array(Pcov)
    Pexact = np.array(Pexact)
    Pu = np.array(Pu)
    Pr = np.array(Pr)
    Pfixed = np.array(Pfixed)
    Pmax = np.array(Pmax)

    # plot of marginal distribution of p-values

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.gca()
    ax1.plot(Ugrid, ECDF(Pcov)(Ugrid), label='Full (exact)', c='red', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pexact)(Ugrid), label='Full (asymptotic)', c='k', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pmax)(Ugrid), label='Max test', c='cyan', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pu)(Ugrid), label=r'Selected 1-sparse, $\sigma$ unknown', c='blue', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pr)(Ugrid), label=r'Selected 1-sparse, $\sigma$ known', c='green', linewidth=5, alpha=0.5)
    ax1.plot(Ugrid, ECDF(Pfixed)(Ugrid), label=r'Fixed 1-sparse, $\sigma$ known', c='yellow', linewidth=5, alpha=0.5)
    ax1.set_xlabel('P-value, $p$', fontsize=20)
    ax1.set_ylabel('ECDF($p$)', fontsize=20)
    ax1.plot([0.05,0.05],[0,1], 'k--')
    ax1.legend(loc='lower right')
    
    # conditional distribution of p-values
    # conditioned on selection choosing correct position and sign

    fig2 = plt.figure(figsize=(8,8))
    hypotheses = np.array(hypotheses, np.bool)
    ax2 = fig2.gca()
    ax2.plot(Ugrid, ECDF(Pcov[hypotheses])(Ugrid), label='Full (exact)', c='red', linewidth=5, alpha=0.5)
    ax2.plot(Ugrid, ECDF(Pexact[hypotheses])(Ugrid), label='Full (asymptotic)', c='k', linewidth=5, alpha=0.5)
    ax2.plot(Ugrid, ECDF(Pu[hypotheses])(Ugrid), label=r'Selected 1-sparse, $\sigma$ unknown', c='blue', linewidth=5, alpha=0.5)
    ax2.plot(Ugrid, ECDF(Pr[hypotheses])(Ugrid), label=r'Selected 1-sparse, $\sigma$ known', c='green', linewidth=5, alpha=0.5)
    ax2.set_xlabel('P-value, $p$', fontsize=20)
    ax2.set_ylabel('ECDF($p$)', fontsize=20)
    ax2.plot([0.05,0.05],[0,1], 'k--')
    ax2.legend(loc='lower right')

    dbn1 = {}
    dbn1['exact'] = Pexact
    dbn1['covtest'] = Pcov
    dbn1['unknown'] = Pu
    dbn1['known'] = Pr
    dbn1['fixed'] = Pfixed
    dbn1['max'] = Pmax
    dbn1['hypotheses'] = hypotheses

    return fig1, fig2, dbn1

def power(n, snr, pos, rho=0.25,
          muval = np.linspace(0,5,51)):

    X, mu, beta = parameters(n, rho, pos)

    # form the correct constraints

    con, initial = constraints(X, pos)

    Z_selection = sample_from_constraints(con, initial, ndraw=4000000, burnin=100000)
    S0 = np.dot(X.T, Z_selection.T).T
    W0 = np.ones(S0.shape[0])
    dfam0 = discrete_family(S0[:,pos], W0)

    one_sided_acceptance_region = dfam0.one_sided_acceptance(0)
    def one_sided_power(mu):
        L, U = one_sided_acceptance_region
        return 1 - (dfam0.cdf(mu,U) - dfam0.cdf(mu, L))

    power_fig = plt.figure(figsize=(8,8))
    power_ax = power_fig.gca()
    power_ax.set_ylabel('Power', fontsize=20)
    power_ax.legend(loc='lower right')
    power_ax.set_xlabel('Effect size $\mu$', fontsize=20)
    full_power = np.array([one_sided_power(m) for m in muval])
    print full_power
    power_ax.plot(muval, full_power, label='Reduced model UMPU', linewidth=7, alpha=0.5)
    power_ax.legend(loc='lower right')
    power_ax.set_xlim([0,5])
    power_ax.plot([snr,snr],[0,1], 'k--')
    print one_sided_power(snr)
    return power_fig, {'full':full_power}

def main():

    power_fig = power(20, 3., 3)[0]
    power_fig.savefig('full_data_power.pdf')

    fig1, fig2, dbn1 = simulation(20, 3., 3, nsim=1000)
    fig1.savefig('reduced_1sparse_20.pdf')
    fig2.savefig('reduced_1sparse_20_cond.pdf')
    np.savez('pval_20.npz', **dbn1)

    fig1, fig2, dbn1 = simulation(100, 3., 3, nsim=1000)
    fig1.savefig('reduced_1sparse_100.pdf')
    fig2.savefig('reduced_1sparse_100_cond.pdf')
    np.savez('pval_100.npz', **dbn1)

if __name__ == "__main__":
    main()
