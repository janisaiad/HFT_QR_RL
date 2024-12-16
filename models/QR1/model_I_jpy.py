# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: EA
#     language: python
#     name: python3
# ---

# %% [markdown]
# # QR MODEL I
#

# %%
import numpy as np


# %% [markdown]
# In this first model the middle price does not moves, meaning the price of the asset does not moves at all!

# %%
def model_I(K, T, lambda_L, lambda_C, lambda_M):
    '''
    K is the number of queues
    T is the number of steps
    lambda_L = [lambda_L_1, ..., lambda_L_K] are the parameters for Limit orders
    where lambda_L_i(n) is the intensity of adding a Limit order at i when there is a queue size of n
    lambda_L are the parameters for Cancellation orders
    lambda_M are the parameters for Market orders
    '''
    ## COMMENT INITIALISER??
    Q = [0 for i in range (K)]
    Q = Q[::-1]+[0 for i in range (K)]
    for i in range (len(Q)//2): # Q of size 2K
        Q[K+i] = np.random.poisson(lam=lambda_L[i](0))
        Q[K-1-i] = np.random.poisson(lam=lambda_L[i](0))
    for i in range (T):
        maj(lambda_L, lambda_C, lambda_M, Q, K)
    return Q

def maj(lambda_L, lambda_C, lambda_M, Q, K):
    '''
    Updating the LOB
    '''
    for i in range (len(Q)//2): # Q of size 2K
        # adding Limit orders, deleting cancellations orders + sell/buy orders
        Q[K+i]= Q[K+i]+np.random.poisson(lam=lambda_L[i](Q[K+i]))-np.random.poisson(lam=lambda_C[i](Q[K+i]))-np.random.poisson(lam=lambda_M[i](Q[K+i]))
        Q[K-i-1]= Q[K-i-1]+np.random.poisson(lam=lambda_L[i](Q[K+i]))-np.random.poisson(lam=lambda_C[i](Q[K+i]))-np.random.poisson(lam=lambda_M[i](Q[K+i]))
    return Q


# %% [markdown]
# We need to have positive values, meaning that we need to have more insertion than cancellation and market orders.

# %%
K = 2
def f_L_1(n):
    if n == 0:
        return 70
    return 50/n

def f_L_2(n):
    if n == 0:
        return 10
    return 5/n

def f_C_1(n):
    if n == 0:
        return 20
    return 30/n

def f_C_2(n):
    if n == 0:
        return 0
    return 4/n

def f_M_1(n):
    if n == 0:
        return 0
    return 20/n

def f_M_2(n):
    if n == 0:
        return 0
    return 1/n

lambda_L = [f_L_1,f_L_2]
lambda_C = [f_C_1,f_C_2]
lambda_M = [f_M_1,f_M_2]

print(lambda_L[0](1))
T = 10
print('State at time',T,':',model_I(K, T, lambda_L, lambda_C, lambda_M))

# %%
