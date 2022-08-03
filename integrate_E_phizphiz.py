import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erf


def integrate_phizphiz(params):
    t, mu, V = params
    
    res = np.zeros(shape=(V.shape[1], V.shape[2]))


    mu1_vec = []
    mu2_vec = []
    s1_vec = []
    s2_vec = []
    rho_vec = []

    for i in range(V.shape[1]):
        for j in range(i+1, V.shape[2]):
            mu1_vec.append(mu[t, i])
            mu2_vec.append(mu[t, j])
            s1_vec.append(np.sqrt(V[t, i, i]))
            s2_vec.append(np.sqrt(V[t, j, j]))
            rho_vec.append(V[t, i, j]/(s1_vec[-1]*s2_vec[-1]))

    mu1_vec = np.array(mu1_vec)
    mu2_vec = np.array(mu2_vec)
    s1_vec = np.array(s1_vec)
    s2_vec = np.array(s2_vec)
    rho_vec = np.array(rho_vec)

    mu_vec = lambda z2: mu1_vec + rho_vec*s1_vec/s2_vec*(z2 - mu2_vec)
    s_vec = np.sqrt(s1_vec**2 * (1-rho_vec**2))

    integrand = lambda z2: 1/np.sqrt(2*np.pi*s2_vec**2)*np.exp(-(z2-mu2_vec)**2/(2*s2_vec**2)) * z2 * ( mu_vec(z2)*(erf(mu_vec(z2)/(np.sqrt(2)*s_vec)) + 1) + np.sqrt(2/np.pi)*s_vec*np.exp(-mu_vec(z2)**2/(2*s_vec**2)) )/2

    res_vec = quad_vec(integrand, 0, np.Inf)[0]
    triu_ind = np.triu_indices(V.shape[1], 1)
    res[triu_ind] = res_vec
    res += res.T

    s_diag_vec = np.sqrt(np.diag(V[t]))
    mu_diag_vec = mu[t]
    res += np.diag((s_diag_vec**2+mu_diag_vec**2)/2 * (erf(mu_diag_vec/(np.sqrt(2)*s_diag_vec)) + 1) + mu_diag_vec*s_diag_vec/np.sqrt(2*np.pi)*np.exp(-mu_diag_vec**2/(2*s_diag_vec**2)))

    return res