# import the necessary modules
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve, minimize, Bounds
import cvxopt


def quad_format_combined(A_re_Y, A_im_Y, Y_re, Y_im, M_Y, lambda_0): 
    
    """
        This function reformats the distribution of capacitive times regression as a quadratic program using both components of the 
        admittance.
        Inputs:
            A_re_Y: discretization matrix for the real part of the admittance
            A_im_Y: discretization matrix for the imaginary part of the admittance
            Y_re: real part of the admittance
            Y_im: imaginary part of the admittance
            M_Y: second-order differentiation matrix based on PWL functions
            lambda_0: regularization parameter
        Outputs:
            matrices H and c used in the quadratic program.        
    """
    
    H = 2*((A_re_Y.T@A_re_Y+A_im_Y.T@A_im_Y)+lambda_0*M_Y)
    H = (H.T+H)/2
    c = -2*(Y_re.T@A_re_Y+Y_im.T@A_im_Y)

    return H, c


def A_re_ad(freq_vec, tau_vec):
    
    """
        This function computes the discretization matrix for the real part of the admittance.
        Inputs:
            freq_vec: vector of EIS frequencies
            tau_vec: vector of characteristic timescales
        Output: discretization matrix for the real part of the admittance.
    """
    
    omega_vec = 2.*np.pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    out_A_re = np.zeros((N_freqs, N_taus))
        
    for p in range(0, N_freqs):
        for q in range(0, N_taus):
            if q == 0:
                out_A_re[p, q] = 0.5*(omega_vec[p]**2*tau_vec[q]**2)/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q])
            elif q == N_taus-1:
                out_A_re[p, q] = 0.5*(omega_vec[p]**2*tau_vec[q]**2)/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q]/tau_vec[q-1])
            else:
                out_A_re[p, q] = 0.5*(omega_vec[p]**2*tau_vec[q]**2)/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_re


def A_im_ad(freq_vec, tau_vec):
    
    """
        This function computes the discretization matrix for the imaginary part of the admittance.
        Inputs:
            freq_vec: vector of EIS frequencies
            tau_vec: vector of characteristic timescales
        Output: discretization matrix for the imaginary part of the admittance.
    """
    
    omega_vec = 2.*np.pi*freq_vec

    N_taus = tau_vec.size
    N_freqs = freq_vec.size

    out_A_im = np.zeros((N_freqs, N_taus))

    for p in range(0, N_freqs):
        for q in range(0, N_taus):
            if q == 0:
                out_A_im[p, q] = 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q])
            elif q == N_taus-1:
                out_A_im[p, q] = 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q]/tau_vec[q-1])
            else:
                out_A_im[p, q] = 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_im


def compute_M_D2(tau_vec):
    
    """
        This function computes the second-order differentiation matrix for the extended vector of the discretized distribution of capacitive 
        times.
        Input:
            tau_vec: vector of characteristic timescales
        Output: second-order differentiation matrix.
    """
    
    N_taus = tau_vec.size # number of collocation points, i.e., the number of timescales
    
    out_M = np.zeros([N_taus, N_taus]) # the size of the matrix M is N_tausxN_taus with N_taus the number of collocation points
    
    out_L_temp = np.zeros((N_taus-2, N_taus))
    
    for p in range(0, N_taus-2):
        delta_loc = np.log(tau_vec[p+1]/tau_vec[p])
            
        if p == 0 or p == N_taus-3:
            out_L_temp[p,p] = 2./(delta_loc**2)
            out_L_temp[p,p+1] = -4./(delta_loc**2)
            out_L_temp[p,p+2] = 2./(delta_loc**2)

        else:
            out_L_temp[p,p] = 1./(delta_loc**2)
            out_L_temp[p,p+1] = -2./(delta_loc**2)
            out_L_temp[p,p+2] = 1./(delta_loc**2)
                
    out_M = out_L_temp.T@out_L_temp
        
    return out_M


def L(tau_vec):
    
    """
        This function computes the second-order differentiation matrix for the vector of the discretized distribution of capacitive 
        times.
        Input:
            tau_vec: vector of characteristic timescales
        Output: second-order differentiation matrix.
    """
    
    N_taus = tau_vec.size
    out_L = np.zeros((N_taus-2, N_taus))
    
    for p in range(0, N_taus-2):

        delta_loc = np.log(tau_vec[p+1]/tau_vec[p])
        
        if p==0 or p == N_taus-3:
            out_L[p,p] = 2./(delta_loc**2)
            out_L[p,p+1] = -4./(delta_loc**2)
            out_L[p,p+2] = 2./(delta_loc**2)
        else:
            out_L[p,p] = 1./(delta_loc**2)
            out_L[p,p+1] = -2./(delta_loc**2)
            out_L[p,p+2] = 1./(delta_loc**2)

    return out_L


def cvxopt_solve_qpr(P, q, G=None, h=None, A=None, b=None):
    
    """
        This function formats numpy matrix to cvxopt matrix before it conducts the quadratic programming with cvxopt and outputs the optimum 
        in numpy array format.
    """
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        
    cvxopt.solvers.options['abstol'] = 1e-10
    cvxopt.solvers.options['reltol'] = 1e-10
    sol = cvxopt.solvers.qp(*args)
    
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))


def Simple_run(Z_exp, freq_vec, tau_vec, A_re_Y, A_im_Y, M_Y, lambda_0):
    
    """
        This function deconvolves the distribution of capacitive times (DCT) using ridge regression
        Inputs:
            Z_exp: impedance spectrum
            freq_vec: vector of EIS frequencies
            tau_vec: vector of characteristic timescales
            A_re_Y: discretization matrix for the real part of the admittance based on piecewise-linear (PWL) functions
            A_im_Y: discretization matrix for the imaginary part of the admittance based on PWL functions
            M_Y: second-order differentiation matrix based on PWL functions
            lambda_0: regularization parameter
        Outputs:
            extended DCT vector, discretized DCT vector, conductance G_inf, capacitance C_0
    """
    
    N_taus = len(tau_vec)
    
    # Step 1: Define the experimental admittance
    Y_exp = 1/Z_exp
    Y_re = Y_exp.real
    Y_im = Y_exp.imag
    
    # Step 2: Conduct ridge regression
    N_RL = 2 # we include the conductance and capacitance components
    
    # Step 2.1: Non-negativity constraint
    lb = np.zeros([N_taus+N_RL])
    bound_mat = np.eye(lb.shape[0])
    
    # Step 2.2: Carry out ridge regression
    H_combined, c_combined = quad_format_combined(A_re_Y, A_im_Y, Y_re, Y_im, M_Y, lambda_0)
    x_RR = cvxopt_solve_qpr(H_combined, c_combined, -bound_mat, lb) 
    
    # Step 3: Recover the DCT, conductance, and capacitance  
    G_inf_RR, C_0_RR = x_RR[0:N_RL]
    gamma_RR = x_RR[N_RL:]
    
    return x_RR, gamma_RR, G_inf_RR, C_0_RR


def count_parameters(model):
    
    """
        This function is used to count the number of parameters of a deep neural network
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Peak separation using either Gaussian functions or the Havriliak-Negami DCT (see (23))

def gauss_fct(p, tau_vec, N_peaks):
    
    """
        This function returns a sum of Gaussian functions, whose parameters, i.e., the prefactor sigma_f, mean mu_log_tau, and 
        standard deviation 1/inv_sigma for each DCT peak, are encapsulated in p
        Inputs:
            p: list of parameters of the Gaussian distributions
            tau: vector of timescales onto which the Gaussian distributions are defined
            N_peaks: number of peaks of the final Gaussian distribution, i.e., number of peaks in the distribution of capacitive times that 
            will be fitted by the sum of Gaussian distributions
        Output:
            Sum of N_peaks Gausian distributions
    """
    
    gamma_out = np.zeros_like(tau_vec)
    
    for k in range(N_peaks):
        
        sigma_f, mu_log_tau, inv_sigma = p[3*k:3*k+3] 
        gaussian_out = sigma_f**2*np.exp(-inv_sigma**2/2*((np.log(tau_vec) - mu_log_tau)**2))
        # note: I did inv_sigma b/c this leads to less computational problems (no exploding gradient when sigma->0)
        gamma_out += gaussian_out 
    return gamma_out  


def HN_fct(p, tau_vec, N_peaks):
    
    """
        This function returns a sum of Havriliak-Negami (HN) functions, whose parameters, i.e., resistance R_ct, the characteristic log 
        timescale log_tau_0, the dispersion parameter phi, and the symmetry parameter phi for each DCT peak, are encapsulated in p
        Inputs:
            p: list of parameters of the HN distributions
            tau: vector of timescales onto which the HN distributions are defined
            N_peaks: number of peaks of the final HN distribution, i.e., number of peaks in the distribution of capacitive times that 
            will be fitted by the sum of HN distributions
        Output:
            Sum of N_peaks HN distributions
    """
    
    gamma_out = np.zeros_like(tau_vec)
    
    for k in range(N_peaks):
        
        R_ct, log_tau_0, phi, psi = p[4*k:4*k+4] 
        
        x = np.exp(phi*(np.log(tau_vec)-log_tau_0))
        
        theta = np.arctan(np.abs(np.sin(np.pi*phi)/(x+np.cos(np.pi*phi))))
        
        num = R_ct*x**psi*np.sin(psi*theta)
        denom = np.pi*(1+np.cos(np.pi*phi)*x+x**2)**(psi/2)
        
        DCT_HN_out = num/denom
        
        gamma_out += DCT_HN_out 
        
    return gamma_out  


def peak_analysis_Gauss(tau_vec, gamma_vec, N_peaks, method = 'separate'):
    
    """
        This function is used to separate the peaks in a distribution of capacitive times (DCT) spectrum using Gaussian functions
        Inputs:
            tau_vec: vector of timescales
            gamma_vec: DCT vector
            N_peaks: number of peaks in the distribution of capacitive times that will be fitted by the sum of Gaussian distributions
            method: either 'separate' -to fit each DCT peak separately- or 'combined' -to fit all the DCT peaks at once-
        Output:
            DCT fit based on a sum of N_peaks Gausian distributions
    """
    
    # Step 1: define the necessary quantities before the subsequent optimizations
    
    # upper and lower log tau values
    log_tau_min = np.min(np.log(tau_vec))
    log_tau_max = np.max(np.log(tau_vec))

    # list of gaussian matrices
    p_fit_list_separate = [0]*N_peaks  # list of p values for separate case
    p_fit_list_combined = [0]*N_peaks  # list of p values for combined case
    lb_list = [0]*N_peaks    # list of lower bounds for the p values of each peak
    ub_list = [0]*N_peaks    # list of upper bounds for the p values of each peak
    
    # list of gaussian matrices
    gamma_fit_list = [0]*N_peaks 
    
    diff_gamma = np.copy(gamma_vec) # difference between DCT and optimized DCT peaks
    
    # DCT fit and parameters
    out_gamma_fit = np.zeros_like(tau_vec)
    out_p_fit = [0]*N_peaks
    
    # Step 2: optimal fit of each DCT peak
    
    for n in range(N_peaks):
        
        # bounds for N_peaks optimizations
        max_sigma_f = 1.3*np.sqrt(np.max(diff_gamma))
        lb = np.array([0, log_tau_min, 1.0/(log_tau_max-log_tau_min)]) # lower bounds for the subsequent minimization
        ub = np.array([max_sigma_f, log_tau_max, np.inf]) # upper bounds for the subsequent minimization
        lb_list[n] = lb
        ub_list[n] = ub
        
        # residual function to minimize for each DCT peak
        residual_fc = lambda p: np.linalg.norm(gauss_fct(p, tau_vec, 1) - diff_gamma) ** 2
        
        # initial guesses for the p values for each DCT peak
        index_diff_peak = np.argmax(diff_gamma)
        log_tau_diff_peak = np.log(tau_vec[index_diff_peak])
        diff_gamma_at_peak = diff_gamma[index_diff_peak]
        sigma_f_at_peak = np.sqrt(diff_gamma_at_peak)
        p_0 = np.array([sigma_f_at_peak, log_tau_diff_peak, 1]) 
        
        # minimization
        options = {'disp': True, 'maxiter': 1e5} # additional options for the optimization
        
        # optimized p values and optimized fit of each DCT peak
        out_fit = minimize(residual_fc, p_0, bounds=list(zip(lb, ub)), method='SLSQP', options=options)
        p_fit = out_fit.x
        gamma_fit = gauss_fct(p_fit, tau_vec, 1)
        
        # save the n-th p values
        p_fit_list_separate[n] = p_fit
        
        # save the n-th gamma fit
        gamma_fit_list[n] = gamma_fit
        
        # initial DCT without the peak that was just fitted
        diff_gamma -= gamma_fit
    
    # Step 3: optimal fit of the complete DCT function
    
    # bounds for the overall optimization
    bnds_list = [] # concatenation of all the p values
    
    for k in range(N_peaks):
        bnds_list.extend([(lb_list[k][0], ub_list[k][0]), (lb_list[k][1], ub_list[k][1]), (lb_list[k][2], ub_list[k][2])])     
    
    # initial values of the parameters for the overall optimization
    p_fit_ini = np.concatenate([p_fit_list_separate[k] for k in range(N_peaks)])
    
    # residual function to minimize
    residual_fct_fit = lambda p: np.linalg.norm(gauss_fct(p, tau_vec, N_peaks) - gamma_vec) ** 2
    
    # minimization
    out_fit_tot = minimize(residual_fct_fit, p_fit_ini, method='SLSQP', options=options, bounds = bnds_list)
    
    # optimized p values and optimized fit
    p_fit_tot = out_fit_tot.x
    gamma_fit_tot = gauss_fct(p_fit_tot, tau_vec, N_peaks)
    
    # separate the optimized p values
    p_fit_list_combined = p_fit_tot
    
    if method == 'separate': # fit separate Gaussians to the DCT spectrum
        out_gamma_fit = [gauss_fct(p_fit_list_separate[n], tau_vec, 1) for n in range(N_peaks)]
        out_p_fit = p_fit_list_separate #np.concatenate([p_fit_list_separate[k] for k in range(N_peaks)])
    
    else:
        out_gamma_fit = gamma_fit_tot
        out_p_fit = p_fit_list_combined
        
    return out_gamma_fit, out_p_fit


def peak_analysis_HN(tau_vec, gamma_vec, N_peaks, method = 'separate'):
    
    """
        This function is used to separate the peaks in a distribution of capacitive times (DCT) spectrum using Havriliak-Negami (HN) 
        functions
        Inputs:
            tau_vec: vector of timescales
            gamma_vec: DCT vector
            N_peaks: number of peaks in the distribution of capacitive times that will be fitted by the sum of Gaussian distributions
            method: either 'separate' -to fit each DCT peak separately- or 'combined' -to fit all the DCT peaks at once-
        Output:
            DCT fit based on a sum of N_peaks HN distributions
    """
    
    # Step 1: define the necessary quantities before the subsequent optimizations
    
    # upper and lower log tau values
    log_tau_min = np.min(np.log(tau_vec))
    log_tau_max = np.max(np.log(tau_vec))

    # list of ZARC matrices
    p_fit_list_separate = [0]*N_peaks  # list of p values for separate case
    p_fit_list_combined = [0]*N_peaks  # list of p values for combined case
    lb_list = [0]*N_peaks    # list of lower bounds for the p values of each peak
    ub_list = [0]*N_peaks    # list of upper bounds for the p values of each peak
    
    # list of ZARC matrices
    gamma_fit_list = [0]*N_peaks 
    
    diff_gamma = np.copy(gamma_vec) # difference between DCT and optimized DCT peaks
    
    # DCT fit
    out_gamma_fit = np.zeros_like(tau_vec)
    out_p_fit = [0]*N_peaks
    
    # Step 2: optimal fit of each DCT peak
    
    for n in range(N_peaks):
        
        # bounds for N_peaks optimizations
        max_sigma_f = 1.3*np.sqrt(np.max(diff_gamma))
        lb = np.array([0, log_tau_min, 0, 0]) # lower bounds for the subsequent minimization
        ub = np.array([np.inf, log_tau_max, 1, 1]) # upper bounds for the subsequent minimization
        lb_list[n] = lb
        ub_list[n] = ub
        
        # residual function to minimize for each DCT peak
        residual_fc = lambda p: np.linalg.norm(HN_fct(p, tau_vec, 1) - diff_gamma) ** 2
        
        # initial guesses for the p values for each DCT peak
        index_diff_peak = np.argmax(diff_gamma)
        log_tau_diff_peak = np.log(tau_vec[index_diff_peak])
        diff_gamma_at_peak = diff_gamma[index_diff_peak]
        sigma_f_at_peak = np.sqrt(diff_gamma_at_peak)
        p_0 = np.array([sigma_f_at_peak, log_tau_diff_peak, 0.5, 0.5]) 
        
        # minimization
        options = {'disp': True, 'maxiter': 1e5} # additional options for the optimization
        
        # optimized p values and optimized fit of each DCT peak
        out_fit = minimize(residual_fc, p_0, bounds=list(zip(lb, ub)), method='SLSQP', options=options)
        p_fit = out_fit.x
        gamma_fit = HN_fct(p_fit, tau_vec, 1)
        
        # save the n-th p values
        p_fit_list_separate[n] = p_fit
        
        # save the n-th gamma fit
        gamma_fit_list[n] = gamma_fit
        
        # initial DCT without the peak that was just fitted
        diff_gamma -= gamma_fit
    
    # Step 3: optimal fit of the complete DCT function
    
    # bounds for the overall optimization
    bnds_list = [] # concatenation of all the p values
    
    for k in range(N_peaks):
        bnds_list.extend([(lb_list[k][0], ub_list[k][0]), (lb_list[k][1], ub_list[k][1]), (lb_list[k][2], ub_list[k][2]), (lb_list[k][3], ub_list[k][3])])     
    
    # initial values of the parameters for the overall optimization
    p_fit_ini = np.concatenate([p_fit_list_separate[k] for k in range(N_peaks)])
    
    # residual function to minimize
    residual_fct_fit = lambda p: np.linalg.norm(HN_fct(p, tau_vec, N_peaks) - gamma_vec) ** 2
    
    # minimization
    out_fit_tot = minimize(residual_fct_fit, p_fit_ini, method='SLSQP', options=options, bounds = bnds_list)
    
    # optimized p values and optimized fit
    p_fit_tot = out_fit_tot.x
    gamma_fit_tot = HN_fct(p_fit_tot, tau_vec, N_peaks)
    
    # separate the optimized p values
    p_fit_list_combined = p_fit_tot
    
    if method == 'separate': # fit separate HNs to the DCT spectrum
        out_gamma_fit = [HN_fct(p_fit_list_separate[n], tau_vec, 1) for n in range(N_peaks)]
        out_p_fit = p_fit_list_separate #np.concatenate([p_fit_list_separate[k] for k in range(N_peaks)])
    
    else:
        out_gamma_fit = gamma_fit_tot
        out_p_fit = p_fit_list_combined
        
    return out_gamma_fit, out_p_fit