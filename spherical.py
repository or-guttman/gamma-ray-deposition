import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.special import roots_legendre

def calc_all_single_component(rho_0, rho_0_rad, t_0, v_space, xi_calculate, M_required):

    # Discretize density profiles and rescale so with the mass (calculated at the given resolution of v_space) is M_required
    rho_0_disc, rho_0_rad_disc = discretize_and_rescale_profiles(rho_0, rho_0_rad, t_0, v_space, M_required)
    
    M_rad = calc_M(rho_0_rad_disc, t_0, v_space)
    column_density_0 = calc_column_density_0(rho_0_disc, rho_0_rad_disc, t_0, v_space, M_rad)
    column_density_t_squared = column_density_0 * t_0**2
    
    f_xi = calc_f_xi(xi_calculate, rho_0_disc, rho_0_rad_disc, t_0, v_space, column_density_0, M_rad)
    def f_xi_interp(xi_requested):
        # This is an interpolated f_xi, given by interpolation on the calculated values at xi_calculate
        values = np.exp(interp1d(np.log(xi_calculate), np.log(f_xi), bounds_error=False, fill_value='extrapolate')(np.log(xi_requested)))
        values[values>1] = 1.0
        return values
        
    f_xi_1 = calc_f_xi([1.0], rho_0_disc, rho_0_rad_disc, t_0, v_space, column_density_0, M_rad)
    n = calc_n(f_xi_1[0])

    def f_gamma_E_t(kappa_gamma_E_t, t):
        t_gamma_E = np.sqrt(kappa_gamma_E_t*column_density_t_squared)
        return f_xi_interp(t/t_gamma_E)
    
    return M_rad, column_density_t_squared, f_xi_interp, n, f_gamma_E_t

def discretize_and_rescale_profiles(rho_0, rho_0_rad, t_0, v_space, M_required):
    # Discretized rho_0 and rho_0_rad
    rho_0_disc = rho_0(v_space)
    rho_0_rad_disc = rho_0_rad(v_space)
        
    # Due to the resolution of v_space, the calculated mass will differ from the required mass
    # Rescale to give M_required at the given resolution
    M_calculated = calc_M(rho_0_disc, t_0, v_space)

    # make sure the v_space is enough for 1% accuracy in M
    if np.abs(M_calculated/M_required-1)>0.01:
        raise Exception("M_calculated and M_required differ by >1%, change v_space: increase resolution/limits")
    
    rho_0_disc = rho_0_disc * M_required / M_calculated
    rho_0_rad_disc = rho_0_rad_disc * M_required / M_calculated

    return rho_0_disc, rho_0_rad_disc
    
def calc_n(f_xi_1):
    # calculates n, the interpolation parameter of the analytic approximation for f_{\gamma}: f_{\gamma,an.} = [1+(t/t_\gamma)^n]^(-2/n)
    # using eq. (27) of Guttman et al 2024
    n = -2 * np.log(2) / np.log(f_xi_1)
    return n

def calc_f_xi(xi, rho_0_disc, rho_0_rad_disc, t_0, v_space, column_density_0, M_rad):
    # calculates f_xi(xi) (eq. 18), that is used as: f_\gamma(E,t) = f_xi(t/t_{\gamma,E}) 

    mu_space, weight_mu = roots_legendre(20) # we use Gauss–Legendre quadrature for the angular integration

    f_xi = np.zeros_like(xi)

    for xi_index, xi_val in enumerate(xi):
        f_from_v = np.zeros_like(v_space) # fraction deposited from source at v
        for v_index, v_val in enumerate(v_space):
            if rho_0_rad_disc[v_index] != 0: # if no radioactive matter at v, no need to calc fraction deposited from there
                f_from_v_and_mu = np.zeros_like(mu_space)  # fraction deposited from source at v to direction mu
                for mu_index, mu_val in enumerate(mu_space):
                    column_density_0_from_v_and_mu = calc_column_density_0_from_v_and_mu(v_val, mu_val, rho_0_disc, t_0, v_space)
                    f_from_v_and_mu[mu_index] = 1 - np.exp(-xi_val**-2 * column_density_0_from_v_and_mu/column_density_0)
                f_from_v[v_index] = 1 / 2 * np.sum(f_from_v_and_mu * weight_mu)
    
        f_xi[xi_index] = trapz(4 * np.pi * v_space**2 * rho_0_rad_disc * t_0 ** 3 * f_from_v, v_space) / M_rad

    return f_xi

def calc_column_density_0(rho_0_disc, rho_0_rad_disc, t_0, v_space, M_rad):
    # Calculates the average column density that the radioactive matter sees, at t_0 (eq. 19)

    mu_space, weight_mu = roots_legendre(20) # we use Gauss–Legendre quadrature for the angular integration

    column_density_0_from_v = np.zeros_like(v_space) # column density that a source at v sees

    for v_index, v_val in enumerate(v_space):
        if rho_0_rad_disc[v_index] != 0: # if no radioactive matter at v, no need to calc column density from there
            column_density_0_from_v_and_mu = np.zeros_like(mu_space) # column density that a source at v sees to direction mu
            for mu_index, mu_val in enumerate(mu_space):
                column_density_0_from_v_and_mu[mu_index] = calc_column_density_0_from_v_and_mu(v_val, mu_val, rho_0_disc, t_0, v_space)
            column_density_0_from_v[v_index] = 1 / 2 * np.sum(column_density_0_from_v_and_mu * weight_mu)

    column_density_0 = trapz(4 * np.pi * v_space**2 * rho_0_rad_disc * t_0 ** 3 * column_density_0_from_v, v_space) / M_rad

    return column_density_0
    
def calc_column_density_0_from_v_and_mu(v_val, mu_val, rho_0_disc, t_0, v_space):
    # Calculates the column density that the radioactive matter sees at v to direction mu, at t_0 (eq. 11)
    c = 29979245800
    u_space = np.arange(0, 2 * c, v_space[1] - v_space[0])  # we integrated u from 0 to infinity, and 2*c*t is the maximal distance between points in the ejecta
    v_along_path = np.sqrt(v_val**2 + u_space**2 + 2 * v_val * u_space * mu_val)
    column_density_0_from_v_and_mu = trapz(interp1d(v_space, rho_0_disc, fill_value=0, bounds_error=False)(v_along_path)*t_0, u_space)
    
    return column_density_0_from_v_and_mu
    
def calc_M(rho_0_disc, t_0, v_space):
    # Calculates mass of profile rho_0(v)
    M = trapz(4 * np.pi * v_space**2 * rho_0_disc * t_0**3, v_space)
    return M