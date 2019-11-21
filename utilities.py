
"""
This class represents a single user arriving to the charging facility 
Author(s):  
            Cesar Santoyo
"""
from prettytable import PrettyTable
from scipy.integrate import quad
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fmin
import math

class User:
    """
        Member functions of the user class for the eve arrival simulation
    """
    
    def __init__(self, a_j, x_j, alpha_j, r_j):
        """
        This function initializes the user class.
        
        Parameters
        ----------
            a_j: list
                arrival time 
            x_j: list
                charging demand
            alpha_j:
                impatience factor
            r_j: 
                charging rates
            
        Returns
        -------
        n/a
        
        Notes
        -----
        n/a
 
        """
        self.a_j = a_j                          # Arrival time(h)
        self.x_j = x_j                          # Charging Demand (kWh)
        self.alpha_j = alpha_j                  # Impatience factor ($/hr.)
        self.r_j = r_j                          # Charging rate (kW)
        self.u_j = self.x_j/self.r_j            # Time to charge
        self.finaltime = self.a_j + self.u_j    # Final time
        
    def get_user_attribute_table(self):
        """
            This function prints out a table of all the user parameters
        Parameters
        ----------
            n/a
  
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
 
        """
        t = PrettyTable(['User Attribute', 'Value'])
        t.add_row(['Arrival Time', self.a_j])
        t.add_row(['Charging Demand', self.x_j])
        t.add_row(['Impatience Factor', self.alpha_j])
        t.add_row(['Rate', self.r_j])
        t.add_row(['Time to Charge', self.u_j])
        print(t)
        
class ChargingFacility:
    """
        Member functions of the charging facility class. Contains functions & variables 
        for EV simulation.
    """
    
    def __init__(self, V, r, T, alpha_list, x_list, lambdaval, num_t_samples):
        """
            This function initializes the user class.
        
        Parameters
        ----------
            V : numpy array
                y-intercept of pricing function
            r : list
                charging rate (i.e., inverse of pricing function slope)
            T: float
                simulation total time (hrs.)
            alpha_list: list
                contains min alpha, max alpha, and alpha's distribution (impatience factor, $/hr)
            xlist: list
                contains min x, max x, and x's distribution (charging demand, kWh)
            lambaval: list
                value of lambda 
            num_t_samples: list
                number of time samples
                
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
             
        """
        
        self.V_coeff = V
        self.r_coeff = r
        self.T = T
        self.num_t_samples = num_t_samples

        # Pull min & max values of demand & impatience
        self.alpha_minmax = np.array(alpha_list[0:2])
        self.x_minmax = np.array(x_list[0:2])
        
        # Distributions STrings Stored
        self.alpha_distribution = alpha_list[2]
        self.x_distribution = x_list[2]
        
        # Compute Probability Function will be min()
        self._compute_probability_of_min()

        # Compute Expectation
        self.exp_uj = self._get_expectation_uj()
        self.lambda_val = lambdaval
        self.exp_eta = lambdaval*self.exp_uj
        self._get_expectation_second_moment_rj()
        
        # Compute Upper Bounds
        self._compute_M_delta()
        self._compute_R_delta()


    def _get_price_func_param(self, x_j, alpha_j):
        """
            This function returns the rate of the function which minimizes the prices for a particular user.
        
        Parameters
        ----------
            x_j: numpy array
            
                user demand (kWh)
                
            alpha_j: numpy array
            
                user impatience factor ($/hr.)
            
        Returns
        -------
            self.r_coeff[indexofmin]: numpy array
                
                rate of cost function which minimizes user j's cost
        
        Notes
        -----
            n/a
         
        """
        store_price = []
        for ii in range(0, len(self.V_coeff)):
            store_price.append(x_j*(self.V_coeff[ii] + alpha_j/self.r_coeff[ii]))
        
        indexofmin = store_price.index(min(store_price))
        
        return self.r_coeff[indexofmin]
    
    def run_simulation(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        self.UserList = []
        self.time_for_plot = np.linspace(0, self.T, self.num_t_samples)
        users_arrived = np.random.poisson(self.lambda_val*self.T)
        time_range = np.sort(self.T*np.random.rand(users_arrived))
        t = np.tile(self.time_for_plot, (users_arrived, 1))     # time range
        self.usermatrix = np.zeros([users_arrived, self.num_t_samples])     
        
        # Get Active Users
        for a_j in time_range:
            # Draw charging demand
            if self.x_distribution == 'uniform':
                user_xj = np.random.uniform(min(self.x_minmax), max(self.x_minmax))
            elif self.x_distribution == 'trunc_normal':
                lower = min(self.x_minmax)
                upper = max(self.x_minmax)
                mu = (upper + lower)/2
                sigma = (upper - mu)*.5
                user_xj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
                
            # Draw impatience factor
            if self.alpha_distribution == 'uniform':
                user_alphaj = np.random.uniform(min(self.alpha_minmax), max(self.alpha_minmax))
            elif self.alpha_distribution == 'trunc_normal':
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2
                sigma = (upper - mu)*.5
                user_alphaj = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)
                
            user_rj = self._get_price_func_param(user_xj, user_alphaj)
            self.UserList.append(User(a_j, user_xj, user_alphaj, user_rj))
                
        time_range = np.tile(time_range, (self.num_t_samples, 1)).transpose()
       
        charge_time = np.empty([])
        user_rate = np.empty([])
        
        # Loop through user list to get charge time & user rate 
        for user in self.UserList:
            charge_time = np.append(charge_time, user.u_j)
            user_rate = np.append(user_rate, user.r_j)
            
        # Erase first zero & make charge_time (user rate) a matrix
        charge_time = np.delete(charge_time, 0)
        user_rate = np.delete(user_rate, 0)
        charge_time =  np.tile(charge_time, (self.num_t_samples, 1)).transpose()
        user_rate = np.tile(user_rate, (self.num_t_samples, 1)).transpose()
        
        # Assign rows & column indices for times where each vechile is in the system
        rows = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[0]
        cols = np.nonzero((t >= time_range) & (t <= time_range + charge_time))[1]
        
        # Define mask
        mask = np.ones(t.shape, dtype=bool)
        mask[rows, cols] = False
        
        self.usermatrix[rows, cols] = 1
        user_rate[mask] = 0 # set times when user not active to zero
        
        self.activeusers =  np.sum(self.usermatrix, axis=0)
        self.totalrate = np.sum(user_rate, axis=0)
        

    def _get_expectation_uj(self):
        """
            Computes the expected value of the time to charge.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            exp_xj/exp_j: numpy array
                
                expectation of u, i.e., expectation of time to charge
        
        Notes
        -----
            n/a
         """
        exp_xj = self._get_expectation_xj()
        exp_rj = self._get_expectation_rj()
         
        return exp_xj/exp_rj
    
    def _get_expectation_xj(self):
        """
            This function computes the expectation .
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         
        a = min(self.x_minmax)
        b = max(self.x_minmax)
        def pdf_X(X, a, b):
            val = X/(b - a)
            return val
        
        exp_xj = quad(pdf_X, min(self.x_minmax), max(self.x_minmax), args=(a,b))[0]
        
        return exp_xj
    
    def _get_expectation_rj(self):
        """
            This computes the expect value of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_rj = self.exp_rj + probability*self.r_coeff[r_index]
            r_index = r_index + 1
             
        return self.exp_rj
    
    def _get_expectation_second_moment_rj(self):
        """
            This computes the second moment of the charging rate, r_j.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        r_index = 0
        self.exp_second_moment_rj = 0
        for probability in self.price_func_min_probability:
            self.exp_second_moment_rj = self.exp_second_moment_rj + probability*self.r_coeff[r_index]**2
            r_index = r_index + 1
                 
    def _compute_probability_of_min(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
        # Initialize delta arrays
        delta_Vcoeff = np.empty([np.size(self.V_coeff), np.size(self.V_coeff)])
        delta_rcoeff = np.empty([np.size(self.r_coeff), np.size(self.r_coeff)])
        
        # Compute the delta's of P_ki and r_ki 
        for ii in range(0,np.size(self.V_coeff)):
            for jj in range(0,np.size(self.V_coeff)):
                    delta_Vcoeff[ii][jj] = self.V_coeff[ii] - self.V_coeff[jj]
                    delta_rcoeff[ii][jj] = 1/self.r_coeff[ii] - 1/self.r_coeff[jj]
        
        delta_fraction = np.empty([np.size(self.V_coeff), np.size(self.V_coeff)])
        
        # Compute the delta's of V_ki and r_ki 
        for ii in range(0,np.size(self.V_coeff)):
            for jj in range(0,np.size(self.V_coeff)):
                    delta_fraction[ii][jj] = delta_Vcoeff[ii][jj]/delta_rcoeff[jj][ii]

        
        if self.alpha_distribution == 'uniform':
            def pdf_A(alpha, a, b):
                val = 1/(b - a)
                return val
                
        # Place min & max alpha values in numpy array        
        alphamin = min(self.alpha_minmax)
        alphamax = max(self.alpha_minmax)
        
        # Define bounds of uniform distribution
        a = alphamin
        b = alphamax
        
        self.price_func_min_probability = np.empty(self.V_coeff.size)
        lowerbound = np.empty(self.V_coeff.size)
        upperbound = np.empty(self.V_coeff.size)
        
        # Loop through all N pricing functions
        for ii in range(0, np.size(self.V_coeff)):
            lower_indices = np.arange(0, ii) # Define indices less than k
            
            if ii != np.size(self.V_coeff) - 1:
                upper_indices = np.arange(ii+1, self.V_coeff.size) # Define indices greater than k
            else:
                upper_indices = np.arange(ii, self.V_coeff.size) # Define indices greater than k

            
            lower_delta_fraction = np.take(delta_fraction[ii], lower_indices)
            upper_delta_fraction = np.take(delta_fraction[ii], upper_indices)

            lower_delta_fraction = lower_delta_fraction[lower_delta_fraction >= 0]
            upper_delta_fraction = upper_delta_fraction[upper_delta_fraction >= 0]
            
            if lower_delta_fraction.size !=0:
                lowerbound[ii] = max(alphamin, np.max(lower_delta_fraction))
            else:
                lowerbound[ii] = alphamin
            
            # Define upper bound of integration
            # if-statement to catch empty "upper_delta_fraction" (i.e., when k = N)
            if upper_delta_fraction.size != 0:
                upperbound[ii] = min(alphamax, np.min(upper_delta_fraction))
            else:
                upperbound[ii] = alphamax

            if self.alpha_distribution == 'uniform':
                self.price_func_min_probability[ii] = max(0.0, quad(pdf_A, lowerbound[ii], upperbound[ii], args=(a,b))[0])
            elif self.alpha_distribution == 'trunc_normal':
                # Lower & upper limits of truncated normal distribution
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2      # Mean
                sigma = (upper - mu)*.5     # Standard deviation
                self.price_func_min_probability[ii] = max(0.0, \
                    stats.truncnorm.cdf((upperbound[ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)-\
                    stats.truncnorm.cdf((lowerbound[ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma))
        self.lowerbound = lowerbound
        self.upperbound = upperbound
    def _compute_M_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
#        self.delta_M_range = np.linspace(1e-2, 1, 100)
#        self.M_func = self.exp_eta + 1/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(1/9 * np.log(np.divide(1, self.delta_M_range))**2 + \
#        2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
#        
#        self.M_func_loose = self.exp_eta + 2/3 * np.log(np.divide(1, self.delta_M_range)) + \
#        np.sqrt(2*self.exp_eta*np.log(np.divide(1, self.delta_M_range)))
        
        self.confidence_delta_M = np.empty([])
        self.script_M = np.linspace(.05, 100, 1000) + self.exp_eta
#        temp_delta_M = 0.0
#        for M_val in self.script_M:s
        numerator = -(self.script_M - self.exp_eta)**2
        denominator = 2*(self.exp_eta + 1/3 * (self.script_M - self.exp_eta))
        
        self.confidence_delta_M  = np.exp(np.divide(numerator, denominator))

    def _compute_R_delta(self):
        """
            This function computes the probability of a function being the min
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """ 
        self.script_R = np.linspace(self.exp_eta*self.exp_rj,2500,20)
  
        self.confidence_delta_R = np.empty([])

        for script_R_values in self.script_R:
            sum_temp_var = 0.0 # temporary variable to perform summation
            
            # Define lower and upper bound of summation
            M_upper = int(np.floor(script_R_values/self.exp_rj))
            M_lower = int(np.ceil(script_R_values/max(self.r_coeff)))
#            M_lower = 0
            
            # Define numerator of and denominator of exponential
            numerator = -(M_upper - self.exp_eta)**2     
            denominator = 2*(self.exp_eta + 1/3 * (M_upper - self.exp_eta))
            
            # Define m values over which to perform sum
            m = np.linspace(M_lower, M_upper, M_upper - M_lower + 1)
            
            exp_numerator = -(script_R_values - m*self.exp_rj)**2
            exp_denominator = 2*(m*self.exp_second_moment_rj + (1/3)*max(self.r_coeff)*(script_R_values - m*self.exp_rj))
            sum_total_probability = np.multiply(np.exp(np.divide(exp_numerator, exp_denominator)), \
                                    stats.poisson.pmf(m, self.exp_eta, loc = 0))
            
            if script_R_values <= self.exp_eta*self.exp_rj:
                sum_temp_var = 1
            else:
                sum_temp_var = np.sum(sum_total_probability) +  np.exp(np.divide(numerator, denominator))
                sum_temp_var= min(1, sum_temp_var)


#            print(sum_temp_var)
            self.confidence_delta_R = np.append(self.confidence_delta_R, sum_temp_var)
#        self.script_R = self.script_R + self.exp_eta*self.exp_rj
        self.confidence_delta_R = np.delete(self.confidence_delta_R, [0], axis=0)
    def get_pricingfunc_plot(self, DistFlag=None):
        """
            This function prints out all the pricing functions and saves them.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         
        """
        
        if DistFlag is None:
            alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
            f = plt.figure()
            for ii in range(0,self.V_coeff.size):
                plt.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$r_' + str(ii + 1) + ' ,V_' + str(ii + 1) + '$' )
                plt.xlabel(r"$\alpha$")
                plt.ylabel(r"$P_i(\cdot, \alpha)$")
                plt.grid(True, which='both')
            
            plt.legend()
            plt.show()
            f.savefig("./plots/pricingfunc.pdf", bbox_inches='tight')
            
        elif DistFlag == 'one':
            alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
            fig1, ax1 = plt.subplots()
            for ii in range(0,self.V_coeff.size):
                ax1.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$r_' + str(ii + 1) + ' ,V_' + str(ii + 1) + '$' )
                ax1.set_xlabel(r"$\alpha$")
                ax1.set_ylabel(r"$g_i(\cdot, \alpha)$")
                plt.grid(True, which='both')
                plt.legend()
            ax1.set_ylim([min(.95*self.V_coeff), 7.5])
            
            if self.alpha_distribution == 'trunc_normal':
                ax2 = ax1.twinx()
                
                # Lower & upper limits of truncated normal distribution
                lower = min(self.alpha_minmax)
                upper = max(self.alpha_minmax)
                mu = (upper + lower)/2      # Mean
                sigma = (upper - mu)*.5     # Standard deviation
                
                # Standardize Normal Distribution and get pdf values & plot
                alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                ax2.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6, label='Trunc. Norm PDF')
                ax2.set_ylim([0, 1])
                ax2.set_ylabel(r'Probability')
                
                # Get percentile of normal distribution of region to shade
                get_min_alpha_percentile = stats.truncnorm.cdf((self.lowerbound[1] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                get_max_alpha_percentile = stats.truncnorm.cdf((self.upperbound[1] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                
                plt.legend()
                plt.show()
                fig1.savefig("./plots/pricingfunc_with_dist.pdf", bbox_inches='tight')
                
        elif DistFlag == 'two':
                chosen_ii = 1
                alpha_range = np.linspace(min(self.alpha_minmax), max(self.alpha_minmax), 50)
                fig1 = plt.figure()
                fig1.subplots_adjust(wspace=.1)
                ax1 = fig1.add_subplot(1, 2, 1)
                for ii in range(0,self.V_coeff.size):
                    ax1.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], label = r'$R^' + str(ii + 1) + ' ,V^' + str(ii + 1) + '$' )
                    ax1.set_xlabel(r"$\alpha$")
                    ax1.set_ylabel(r"$g_\ell(x, \cdot)$")
                    plt.grid(True, which='both')
                    plt.legend()
                ax1.set_ylim([min(.95*self.V_coeff), 7.5])
                
                if self.alpha_distribution == 'trunc_normal':
                    ax2 = ax1.twinx()
                    
                    # Lower & upper limits of truncated normal distribution
                    lower = min(self.alpha_minmax)
                    upper = max(self.alpha_minmax)
                    mu = (upper + lower)/2      # Mean
                    sigma = (upper - mu)*.5     # Standard deviation
                    
                    # Standardize Normal Distribution and get pdf values & plot
                    alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                    ax2.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6)
                    ax2.set_ylim([0, 1])
                    ax2.axis('off')
                
                ax3 = fig1.add_subplot(1, 2, 2)
                # Plot
                for ii in range(0,self.V_coeff.size):
                    if ii != chosen_ii:
                        ax3.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii],\
                                 color='gray', alpha=.3)
                    else:
                        ax3.plot(alpha_range, self.V_coeff[ii] + alpha_range/self.r_coeff[ii], color='C1')
                ax3.set_xlabel(r'$\alpha$')
                ax3.grid(True)
                ax3.set_ylim([min(.95*self.V_coeff), 7.5])
                ax3.tick_params(axis='y', which='major',labelleft=False)
                
                if self.alpha_distribution == 'trunc_normal':
                        ax4 = ax3.twinx()
                        
                        # Standardize Normal Distribution and get pdf values & plot
                        alpharange_pdf = np.linspace(stats.truncnorm.ppf(0.00, (lower - mu) / sigma , (upper - mu) / sigma), stats.truncnorm.ppf(1.00, (lower - mu) / sigma , (upper - mu) / sigma), 100)
                        ax4.plot(sigma*(alpharange_pdf  + mu/sigma), stats.truncnorm.pdf(alpharange_pdf , (lower - mu) / sigma , (upper - mu) / sigma), 'r--', lw=1.5, alpha=0.6, label=r'$f_A(\alpha)$')
                        ax4.set_ylim([0, 1])
                        ax4.set_ylabel(r'Probability')
                        
                        # Get percentile of normal distribution of region to shade
                        get_min_alpha_percentile = stats.truncnorm.cdf((self.lowerbound[chosen_ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                        get_max_alpha_percentile = stats.truncnorm.cdf((self.upperbound[chosen_ii] - mu)/sigma, (lower - mu) / sigma , (upper - mu) / sigma)
                        
                        # Create range for region to shade
                        alpharange_pdf_shade = np.linspace(stats.truncnorm.ppf(\
                                        get_min_alpha_percentile, (lower - mu) / sigma ,\
                                        (upper - mu) / sigma), stats.truncnorm.ppf(get_max_alpha_percentile,\
                                        (lower - mu) / sigma , (upper - mu) / sigma), 100)
                        # Fill probability Distribution between integration bounds
                        plt.fill_between(sigma*(alpharange_pdf_shade + mu/sigma),\
                                         stats.truncnorm.pdf(alpharange_pdf_shade, (lower - mu) / sigma ,\
                                        (upper - mu) / sigma), color='r', alpha=.1)

                        plt.legend()
                        plt.show()
                fig1.savefig("./plots/pricingfunc_with_dist_twoplot.pdf", bbox_inches='tight')

    def get_sim_plots(self):
        """
            This function runs the charging facility simulation.
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
         """
         
        
        f = plt.figure()
        plt.plot(self.time_for_plot,self.activeusers)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\eta(t)$")
        plt.title(r"Number of users versus Time (hr.)")
        plt.show()
        f.savefig("./plots/num_of_users_vs_t.pdf", bbox_inches='tight')
        
        f = plt.figure()
        plt.plot(self.time_for_plot, self.totalrate)
        plt.grid('True')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$R(t)$")
        plt.title(r"Charging Rate (kW) versus Time (hr.)")
        plt.show()
        f.savefig("./plots/chargerate_vs_t.pdf", bbox_inches='tight')
        
    def get_upperbound_plot(self, N_percentile_store=None, percentiles=None, R_percentile_store=None, percentiles2=None):
        """
            Plots the results from the 
        
        Parameters
        ----------
            n/a
            
        Returns
        -------
            n/a
        
        Notes
        -----
            n/a
        """
        if N_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(1 - self.delta_M_range, self.M_func)
            plt.plot(1 - self.delta_M_range, self.M_func_loose)
            plt.grid('True')
            plt.xlabel(r'1 - $\delta(\mathcal{M})$')
            plt.ylabel(r'$\mathcal{M}$')
            plt.ylim([0, 1.05])
            plt.show()
        else:
            N_percentile_store = np.delete(N_percentile_store, [0], axis=0)
            N_percentile_means = np.mean(N_percentile_store, axis=0)
#            N_percentile_std = np.std(N_percentile_store, axis=0)
            xerr = [N_percentile_store.min(axis=0)-N_percentile_means, N_percentile_means-N_percentile_store.max(axis=0)]
            f = plt.figure(0)
            plt.plot(self.script_M, 1- self.confidence_delta_M, label=r'$\mathcal{M}$')
#            plt.plot(1 - self.delta_M_range, self.M_func_loose)
            plt.errorbar(N_percentile_means, percentiles/100, xerr=xerr, label=r'$\eta(t)$ Monte Carlo Percentiles')
            plt.grid('True')
            plt.ylabel(r'1 - $\delta(\mathcal{M})$')
            plt.xlabel(r'$\mathcal{M}$')
            plt.xlim([0, max(self.script_M)])
            plt.ylim([0, 1.05])
            plt.title(r'Total Active Users vs. Confidence Interval')
            plt.legend()
            plt.show()
            f.savefig("./plots/errorbars_N_t.pdf", bbox_inches='tight')

        if R_percentile_store is None:
            fig = plt.figure(0)
            plt.plot(self.script_R, 1 - self.confidence_delta_R, label=r'$\mathcal{R}(\delta_\mathcal{R})$')
#            plt.plot(1 - self.delta_R_range, self.M_func_loose)
            plt.grid('True')
            plt.ylabel(r'1 - $\gamma_\mathcal{R}$')
            plt.xlabel(r'$\mathcal{R}(\delta_\mathcal{R})$')
            plt.ylim([0, 1.05])
            plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.show()
        else:
            R_percentile_store = np.delete(R_percentile_store, [0], axis=0)
            R_percentile_means = np.mean(R_percentile_store, axis=0)
            xerr = [R_percentile_store.min(axis=0)-R_percentile_means, R_percentile_means-R_percentile_store.max(axis=0)]
            f = plt.figure(0)
            plt.plot(self.script_R, 1 - self.confidence_delta_R, label=r'$\mathcal{R}$')
            plt.errorbar( R_percentile_means, percentiles2/100, xerr=xerr, label=r'$Q(t)$ Monte Carlo Percentiles')
            plt.grid('True')
            plt.ylabel(r'1 - $\gamma(\mathcal{R})$')
            plt.xlabel(r'$\mathcal{R}$')
            plt.ylim([0, 1.05])
            plt.xlim([0, max(self.script_R)])
            plt.title(r'Total Charging Rate vs. Confidence Interval')
            plt.legend()
            plt.show()
            f.savefig("./plots/errorbars_Q_t.pdf", bbox_inches='tight')