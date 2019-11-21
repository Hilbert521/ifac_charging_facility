"""
Main file to execute simulation from IFAC paper
Author(s):  
            Cesar Santoyo
"""
import utilities as util
import numpy as np
import os
import matplotlib.pyplot as plt 
import scipy.io as sio
from matplotlib import rc

PATHFLAG = 1
# Add Latex to path (Otherwise, render using python)
if os.path.isdir("/usr/local/texlive/2019/bin/x86_64-darwin") and PATHFLAG==1:
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'
    # Set Render &  Compile Settings
    rc('text', usetex=True)
    rc('font', family='serif')
    PATHFLAG = 0
elif os.path.isdir("/usr/share/texlive/texmf-dist") and PATHFLAG==1:
    os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist'
    # Set Render &  Compile Settings
    rc('text', usetex=True)
    rc('font', family='serif')
    PATHFLAG = 0
else:
   print('This code uses latex to render plots. Please add your latex installation path to python.')
    
# Create Directory for plots
if not os.path.exists('./plots'):
    os.makedirs('./plots')

  
T = 100;                    # Total simulation time (hr)
num_t_samples = T*60        # Number of time samples in simulation interval   

# Define upper & lower bounds on impatiance factor (alpha) & distribution type
alpha_min = 10
alpha_max = 50
alpha_dist_type = 'trunc_normal'
alpha_list = [alpha_min, alpha_max, alpha_dist_type]

# Define upper & lower bounds on demand values & distribution type
x_min = 10
x_max = 100 
x_dist_type = 'uniform'
x_list = [x_min, x_max, x_dist_type]

# V & r coefficients for N pricing functions
V_coeff = np.array([5.1, 5.2, 5.3, 5.4, 5.5]) # $/kWh
r_coeff = np.array([ 25.,  30.,  35.,  40., 45.])#kW

# Poisson Parameter
lambdaval = 20 # EVs/hr

# Setup Monte Carlo
mcdraws = 5000
percentiles = np.arange(1,99,6)
N_percentile_store = np.empty([1, len(percentiles)])
R_percentile_store = np.empty([1, len(percentiles)])

for draw in range(0, mcdraws):
    ChargingFacility = util.ChargingFacility(V_coeff, r_coeff, T, alpha_list, x_list, lambdaval, num_t_samples)
    ChargingFacility.run_simulation()
    #ChargingFacility.get_pricingfunc_plot()
    #ChargingFacility.get_upperbound_plot()
    #ChargingFacility.get_sim_plots()
    print("Draw:", draw)
    N_percentile_store = np.append(N_percentile_store, [np.percentile(ChargingFacility.activeusers, percentiles)], axis=0)
    R_percentile_store = np.append(R_percentile_store, [np.percentile(ChargingFacility.totalrate, percentiles)], axis=0)


ChargingFacility.get_upperbound_plot(N_percentile_store, percentiles, R_percentile_store, percentiles)
#R_percentile_store = np.delete(R_percentile_store, [0], axis=0)
#R_percentile_means = np.mean(R_percentile_store, axis=0)
#R_percentile_std = np.std(R_percentile_store, axis=0)


#plt.figure()
#plt.plot(R_percentile_means)

print("Probability of choosing each pricing function: ", ChargingFacility.price_func_min_probability, \
      "\n Sum of Probability", sum(ChargingFacility.price_func_min_probability))
print(r"E[u]: ", ChargingFacility.exp_uj)
ChargingFacility.get_pricingfunc_plot(DistFlag='two')

#plt.plot( 1 - ChargingFacility.confidence_delta_R, ChargingFacility.script_R)
#plt.grid('True')
#plt.show()
#################################################################################
T = 100;                    # Total simulation time (hr)
num_t_samples = T*60        # Number of time samples in simulation interval   

# Define upper & lower bounds on impatiance factor (alpha) & distribution type
alpha_min = 10
alpha_max = 50
alpha_dist_type = 'trunc_normal'
alpha_list = [alpha_min, alpha_max, alpha_dist_type]

# Define upper & lower bounds on demand values & distribution type
x_min = 10
x_max = 100 
x_dist_type = 'uniform'
x_list = [x_min, x_max, x_dist_type]

V_coeff = np.array([5.2, 5.4]) # $/kWh
r_coeff_1 = np.array([ 30., 40.])#kW
#r_coeff_1 = np.array([ 1., 2.])#kW
r_coeff_2 = np.array([ 50., 70.])#kW

# Poisson Parameter
lambdaval = 20 # EVs/hr
ChargingFacility1 = util.ChargingFacility(V_coeff, r_coeff_1, T, alpha_list, x_list, lambdaval, num_t_samples)
ChargingFacility2 = util.ChargingFacility(V_coeff, r_coeff_2, T, alpha_list, x_list, lambdaval, num_t_samples)

fig1 = plt.figure()
fig1.subplots_adjust(wspace=.35)
fig1.subplots_adjust(hspace=.35)

ax1 = fig1.add_subplot(2, 2, 2)
ax1.plot(ChargingFacility1.script_M, 1 - ChargingFacility1.confidence_delta_M,label=r'$\delta(\mathcal{M})$', color='blue')
ax1.plot(ChargingFacility2.script_M, 1 - ChargingFacility2.confidence_delta_M,label=r'$\delta^+(\mathcal{M})$',color='red',linestyle='-.')
ax1.set_ylabel(r"$1-\delta(\mathcal{M})$")
ax1.set_xlabel(r"$\mathcal{M}$")
ax1.grid('True')
ax1.set_ylim([0, 1.1])
ax1.legend()
#ax1.xlim([0, 60])

ax2 = fig1.add_subplot(2, 2, 4)
ax2.plot(ChargingFacility1.script_R, 1 - ChargingFacility1.confidence_delta_R,label=r'$\gamma(\mathcal{R})$', color='blue')
ax2.plot(ChargingFacility2.script_R, 1 - ChargingFacility2.confidence_delta_R,label=r'$\gamma^+(\mathcal{R})$', color='red',linestyle='-.')
ax2.set_ylabel(r"$1 - \gamma(\mathcal{R})$")
ax2.set_xlabel(r"$\mathcal{R}$")
ax2.set_ylim([0, 1.1])
ax2.grid('True')
ax2.legend()
#ax1.xlim([0, 60])


ax3 = fig1.add_subplot(1, 2, 1)

alpha_range = np.linspace(min(ChargingFacility1.alpha_minmax), max(ChargingFacility1.alpha_minmax), 50)
#for ii in range(0,ChargingFacility1.V_coeff.size):
ax3.plot(alpha_range, ChargingFacility1.V_coeff[0] + alpha_range/ChargingFacility1.r_coeff[0], \
             label = r'$R^' + str(0 + 1) + ' ,V^' + str(0 + 1) + '$', color='blue',linestyle='-.')
ax3.plot(alpha_range, ChargingFacility1.V_coeff[1] + alpha_range/ChargingFacility1.r_coeff[1], \
             label = r'$R^' + str(1 + 1) + ' ,V^' + str(1 + 1) + '$', color='blue')
ax3.plot(alpha_range, ChargingFacility2.V_coeff[0] + alpha_range/ChargingFacility2.r_coeff[0], \
             label = r"$(R^1)^+ ,V^" + str(0 + 1) + "$", color='red',linestyle='-.')
ax3.plot(alpha_range, ChargingFacility2.V_coeff[1] + alpha_range/ChargingFacility2.r_coeff[1], \
             label = r"$(R^2)^+ ,V^" + str(1 + 1) + "$", color='red')
ax3.set_xlabel(r"$\alpha$")
ax3.set_ylabel(r"$g_\ell(x, \cdot)$")
ax3.grid(True, which='both')
    
ax3.legend()
plt.show()
fig1.savefig("./plots/illustration_theorem.pdf", bbox_inches='tight')

sio.savemat('gamma_R.mat', {'gamma_R':ChargingFacility.confidence_delta_R, 'R_val':ChargingFacility.script_R})