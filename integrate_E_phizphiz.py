import numpy as np
from scipy.integrate import quad
from scipy.special import erf


def integrand(zi, zj, mui, muj, si, sj, rho):
    di = zi - mui
    dj = zj - muj
    return 1/(2*np.pi*si*sj*np.sqrt(1-rho**2)) * np.exp(-1/(2*(1-rho**2)) * (di**2/si**2 - 2*rho*di*dj/(si*sj) + dj**2/sj**2) ) * zi * zj

def integral_z1(z2, mu1, mu2, s1, s2, rho):
    mu_tilde = mu1 + rho*s1/s2*(z2-mu2)
    s_tilde = s1**2 * (1-rho**2)
    return 1/2 * (mu_tilde*erf(mu_tilde/(np.sqrt(2)*s_tilde)) + np.sqrt(2/np.pi)*s_tilde*np.exp(-mu_tilde**2/(2*s_tilde**2)) + mu_tilde)

def integrand_phizphiz(z2, mu1, mu2, s1, s2, rho):
    gaussian = 1/np.sqrt(2*np.pi*s2**2)*np.exp(-1/(2*s2**2)*(z2-mu2)**2)
    return gaussian * z2 * integral_z1(z2, mu1, mu2, s1, s2, rho)

def integrate_phizphiz(params):
    t, mu, V = params
    
    res = np.zeros(shape=(V.shape[1], V.shape[2]))

    for i in range(V.shape[1]):
        mui = mu[t, i]
        si = np.sqrt(V[t, i, i])

        res[i, i] = (si**2+mui**2)/2 * (erf(mui/(np.sqrt(2)*si)) + 1) + mui*si/np.sqrt(2*np.pi)*np.exp(-mui**2/(2*si**2))

    for i in range(V.shape[1]):
        for j in range(i+1, V.shape[2]):
            mu1 = mu[t, i]
            mu2 = mu[t, j]
            s1 = np.sqrt(V[t, i, i])
            s2 = np.sqrt(V[t, j, j])
            rho = V[t, i, j]/(s1*s2)

            res[i, j] = quad(integrand_phizphiz, 0, np.Inf, args=(mu1, mu2, s1, s2, rho))[0]
            res[j, i] = res[i, j]

    return res