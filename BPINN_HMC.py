#!/usr/bin/env python
# coding: utf-8

# #### Bayesian Physics-Informed Neural Network (with ODE)

# In[1]:

#importing necessary files
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import load_data
import bnn
import bayesian
import hmc
import time
from sklearn.metrics import mean_squared_error
wd = os.path.abspath(os.getcwd())
filename = os.path.basename(__file__)[:-3]

path2saveResults = 'Results/'+filename
if not os.path.exists(path2saveResults):
    os.makedirs(path2saveResults)
#Physical constants
#k=float(input("Spring constant="))
k=5
#A=float(input("Area="))
A=10
#m=float(input("mass"))
m=5
#alpha=float(input("alpha"))
alpha=10
#P0=float(input("P0="))
P0=1
#P1=float(input("P1="))
P1=1/2
#Omega=float(input("Omega"))
Omega=3
#y0=float(input("y(0)"))
y0=2
#ydash0=float(input("y'(0)"))
ydash0=1
c=(A**2)/(alpha*m)
omega2=k/m
 
    
# nmax = 2000 b
#%% Plotting loading
ColorS = [0.5, 0.00, 0.0]
ColorE = [0.8, 0.00, 0.0]
ColorI = [1.0, 0.65, 0.0]
ColorR = [0.0, 1.00, 0.7]

color_mu = 'tab:blue'
color_k = 'tab:red'
color_b = 'tab:green'
#Function for plotting the results
def plotting(N, T_data, X_data, T_exact, X_exact, T_r, mu, std):
    plt.figure(figsize=(1100/72,400/72))
    
    plt.scatter(T_data[:N, :].numpy().flatten(), X_data[:N, :].numpy().flatten(),edgecolors=(0.0, 0.0, 0.7),marker='.',facecolors='none',s=100, lw=1.0,zorder=3, alpha=1.0, label=r'Training data') 

    plt.scatter(T_exact[::2], X_exact[::2], s=100,linewidth=1.0, marker='x',color=ColorI, label='True')

    
    plt.fill_between(T_r.numpy().flatten(), (mu+1.96*std).flatten(), (mu-1.96*std).flatten(), zorder=1, alpha=0.8, color=ColorR)
    plt.plot(T_r.numpy().flatten(), mu.flatten(), color=(0.7, 0.0, 0.0),linewidth=3.0,linestyle="--", zorder=3, alpha=1.0,label=r'B-PINN (mu +- 2std)')


    #plt.yticks([0.3,0.9],fontsize=16)
    #plt.xticks([0.0,0.2,0.6,0.8],fontsize=16)
    plt.legend(loc='upper right',ncol=1, fancybox=True, framealpha=0.,fontsize=24)
    #plt.ylim([0.25,1])
    plt.xlabel("t",fontsize=24)
    plt.ylabel("x(t)",fontsize=24)
    plt.tight_layout()
    plt.savefig(path2saveResults+'/PINN_data_BPINN_.pdf') 
    plt.show()


#%% load data

    
# 
T_data, X_data, T_r, T_exact, X_exact = load_data.load_data()
plt.plot(T_data,X_data)
print('Tensorflow version: ', tf.__version__)
print('Tensorflow-probability version: ', tfp.__version__)

# end
# In[2]:


# define ODE
def ode_fn(t, fn, additional_variables):
    mu, k, b = additional_variables
    with tf.GradientTape() as g_tt:
        g_tt.watch(t)
        with tf.GradientTape() as g_t:
            g_t.watch(t)
            y = fn(t)
        y_der = g_t.gradient(y, t)
    y_dder = g_tt.gradient(y_der, t)
    f = (y_dder) + (c*y_der) + (y*omega2) - ((A*P0/m)+((A*P1*tf.sin(Omega*t))/m))

    return f


    
# #### Prior distributions and change of variables
# 
# We define the prior distributions for $\mu, k, b$ to be independent LogNormal distributions:
# $$\log \mu \sim N(\log 2.2, 0.5) \\\log k \sim N(\log 350, 0.5), \\\log b \sim N(\log 0.56, 0.5)$$
# 
# In practice, we instead sample $\log\mu - \log 2.2, \log k - \log 350$ and $\log b -\log 0.56$, so that those three quantities are independent and identically standard normal random variables. Suppose we have one posterior sample of those three quntities, $\epsilon_\mu, \epsilon_k, \epsilon_b$, then to obtain posterior sample of $\mu, k, b$, we do the following:
# $$\mu = e^{\log 2.2 + \epsilon_\mu} = 2.2 e^{\epsilon_\mu}, k = 350e^{\epsilon_k}, b = 0.56e^{\epsilon_b}.$$
# 
# In this way, re-scaling is done and positivities are guaranteed.

# In[3]:
# N = 150 # 225, 200, 150, 100

# X_data = X_un_tf


def main(N):

    # create Bayesian neural network
    BNN = bnn.BNN(layers=[1,50,50,50, 1])
    # specify number of observations
    
    # specify noise level for PDE
    noise_ode = (0.25,0.25)
    # specify noise level for observations
    noise_u = 0.25
    # create Bayesian model
    model = bayesian.PI_Bayesian(
        x_u=T_data[:N,:],
        y_u=X_data[:N,:],
        # y_u=X_un_tf[:N, :],
        x_pde=T_r,
        pde_fn=ode_fn,
        L=4,
        noise_u=noise_u,
        noise_pde=noise_ode,
        prior_sigma=1,y0=y0,ydash0=ydash0
    )
    # compute log posterior density function
    log_posterior = model.build_posterior(BNN.bnn_fn)
    # create HMC (Hamiltonian Monte Carlo) sampler
    hmc_kernel = hmc.AdaptiveHMC(
        target_log_prob_fn=log_posterior,
        init_state=BNN.variables+model.additional_inits,
        num_results=4000,
        num_burnin=4000,
        num_leapfrog_steps=50,
        step_size=0.001,
    )
    # In[4]:
    
    
    # sampling
    start_time = time.perf_counter()
    samples, results = hmc_kernel.run_chain()
    Acc_rate = np.mean(results.inner_results.is_accepted.numpy())
    print('Accepted rate: ', Acc_rate)
    print(results.inner_results.accepted_results.step_size[0].numpy())
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))
    
    u_pred = BNN.bnn_infer_fn(T_r, samples[:8])
    
    mu = tf.reduce_mean(u_pred, axis=0).numpy()
    std = tf.math.reduce_std(u_pred, axis=0).numpy()

    return model, samples, u_pred, mu, std, Acc_rate

# In[5]:


# compute posterior samples of x and store the results
# x_samples = BNN.bnn_infer_fn(T_r, samples[:2*model.L]).numpy()
# sio.savemat(
#     'results/out_{}.mat'.format(str(N)), {'x_samples': x_samples, 't': T_r.numpy()}
# )


# #### Posterior estimate on function



if __name__ == '__main__':
    N=80
    model, samples, u_pred, mu, std, Acc_rate = main(N)
    plotting(N, T_data, X_data, T_exact, X_exact, T_r, mu, std)
    
    

# In[6]:

"""


log_mu, log_k, log_b = samples[-3:]
mu, k, b = tf.exp(log_mu+model.log_mu_init), tf.exp(log_k+model.log_k_init), tf.exp(log_b+model.log_b_init)

"""



# In[7]:
"""
    def PlotHist(ax, prior, post, var, color, limY, limX, limX0):
    
    ax.hist(post, bins=num_bins, density=True, label='posterior of '+var, color=color, alpha=0.7)
    mean = np.round(np.mean(post),3)
    std = np.round(np.std(post),2)
    ax.set_ylim([0,limY])
    ax.set_xlim([limX0,limX])
    plt.title(var+' = '+str(mean)+'$\pm$'+str(std),fontsize=24, color=color)
    legend = plt.legend(loc='upper right',fontsize=24,ncol=1, fancybox=True, framealpha=0.)
    plt.setp(legend.get_texts(), color=color)
    





num_bins = 30



fig = plt.figure(figsize=(1200/72,800/72))
gs = fig.add_gridspec(2, 2)



s = model.additional_priors[0].sample(3000)
prior = (tf.exp(s + model.log_mu_init)).numpy()
post = mu.numpy()
var = '$c$'
ax1 = plt.subplot(gs[0, 0])


PlotHist(ax1, prior, post, var, color_mu,20, 20, 0.0)



s = model.additional_priors[1].sample(3000)
prior = (tf.exp(s + model.log_k_init)).numpy()
post = k.numpy()
var = '$k$'
ax2 = plt.subplot(gs[0, 1])


PlotHist(ax2, prior, post, var, color_k, 20, 20, 100)




s = model.additional_priors[2].sample(3000)
prior = (tf.exp(s + model.log_b_init)).numpy()
post = b.numpy()
var = '$x_0$'
ax3 = plt.subplot(gs[1, 0])


PlotHist(ax3, prior, post, var, color_b, 23, 20, 0.4)

plt.savefig(path2saveResults+'/BPINN_Para_v2.pdf')
"""
