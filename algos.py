
##############################################################
#                                                            #
#                   Resampling Algorithms                    #
#                                                            #
#                       Vicke Noren                          #
#                           2024                             #
#                                                            #
##############################################################
# 
# This file covers the following resampling algorithms:
# - Residual resampling (RR)
# - Systematic resampling (SR)
# - Residual systematic resampling (RSR)
# 
# They are implemented in standard, matrix, and
# vectorized form.
# 
# The file also provides an example of how resampling works.
# (1) A grid is created.
# (2) Each grid point is evaluated assuming
#     standard normal distribution.
# (3) Particles are resampled using the pdf as weights.
# 
# Implementations in R and MATLAB are also available.
# 
##############################################################

import numpy as np
import pandas as pd
from scipy import stats
from time import process_time_ns

##############################################################
#         Resampling algorithms   -   standard form          #
##############################################################

# Crude resampling
def crude(x, Nout, w):
	return np.sort(np.random.choice(x, Nout, w))

# Systematic resampling (SR)
def sr(w, Nin, Nout, u):
    r = np.zeros(Nin, dtype=int)
    u = u / Nout
    s = 0
    for n in np.arange(Nin):
        k = 0
        s = s + w[n]
        while s > u:
            k = k + 1
            u = u + 1 / Nout
        r[n] = k
    return r

# Residual systematic resampling (RSR)
def rsr(w, Nin, Nout, u):
    r = np.zeros(Nin, dtype=int)
    u = u / Nout
    for n in np.arange(Nin):
        r[n] = np.floor(Nout * (w[n] - u)) + 1
        u = u + r[n] / Nout - w[n]
    return r

# Residual resampling (RR)
def rr(w, Nin, Nout, u):
    r = np.zeros(Nin, dtype=int)
    Nr = Nout
    wr = np.empty((1, Nin,)).flatten() * np.nan
    for n in np.arange(Nin):
        r[n] = np.floor(Nout * w[n])
        wr[n] = Nout * w[n] - r[n]
        Nr = Nr - r[n]
    if Nr > 0:  # Remainder step
        for n in np.arange(Nin):
            wr[n] = wr[n] / Nr
        rres = rsr(wr, Nin, Nr, u)  # Resample with SR or RSR
        for n in np.arange(Nin):
            r[n] = r[n] + rres[n]
    return r

##############################################################
#         Resampling algorithms   -   matrix form            #
##############################################################

# Matrix form of SR
def sr_mat(w, Nin, Nout, u):
    return None

# Matrix form of RSR
def rsr_mat(w, Nin, Nout, u):
    A = np.tril(np.ones((Nin, Nin), dtype=int))
    v = np.hstack((u, np.zeros(Nin - 1)))
    r = np.matmul(np.linalg.inv(A), np.ceil(np.matmul(A, (Nout * w - v)))).astype(int)
    return r

# Matrix form of RR
def rr_mat(w, Nin, Nout, u):
    A = np.tril(np.ones((Nin, Nin), dtype=int))
    Nr = Nout - np.sum(np.floor(Nout * w))
    v = np.hstack((u / Nr, np.zeros(Nin - 1)))
    r = (np.floor(Nout * w) + np.matmul(np.linalg.inv(A), np.ceil( (Nout - np.sum(np.floor(Nout * w))) * np.matmul(A, ((Nout * w - np.floor(Nout * w)) / np.sum(Nout * w - np.floor(Nout * w)) - v)) ) ) ).astype(int)
    return r

##############################################################
#         Resampling algorithms   -   vectorized form        #
##############################################################

# Vectorized version of SR
def sr_vec(w, Nin, Nout, u):
    r = np.linspace(0, (Nout - 1) / Nout, Nout) + u / Nout
    r = np.digitize(r, np.hstack((0, np.cumsum(w))))
    r = np.bincount(r - 1, minlength=Nin)
    return r

# Vectorized version of RSR
def rsr_vec(w, Nin, Nout, u):
    r = Nout * w
    r[0] = r[0] - u
    r = np.ceil(np.cumsum(r)).astype(int)
    r[1:Nin] = np.diff(r)
    return r

# Vectorized version of RR
def rr_vec(w, Nin, Nout, u):
    r = np.floor(Nout * w).astype(int)
    wr = Nout * w - r
    Nr = Nin - np.sum(r)
    if Nr > 0:  # Remainder step
        r = r + rsr_vec(wr / Nr, Nin, Nr, u)
    return r

##############################################################
#                  Conversion algorithms                     #
##############################################################

# Convert vector of replication factors to index vector (curde method)
def indexvector(r, Nin):
    iv = np.zeros(np.sum(r), dtype=int)
    k = 0
    b = iv.copy()
    a = iv.copy()
    c = iv.copy()
    d = iv.copy()
    for n in np.arange(len(r)):
        if r[n] > 0:
            d[k] = n + 1
            if k == 0:
                a[k] = r[n]
            else:
                a[k] = a[k-1] + r[n]
            b[a[k] - r[n]] = 1
            k = k + 1
    c[0] = b[0] - 1
    iv[0] = d[c[0]]
    for n in np.arange(1, np.sum(r)):
        c[n] = c[n - 1] + b[n]
        iv[n] = d[c[n]]
    return iv

# Convert vector of replication factors to index vector (vectorized method)
def indexvector_vec(r, Nin):
    t = r > 0
    a = np.cumsum(r[t])
    b = np.zeros(a[-1], dtype=int)
    b[a - r[t]] = 1
    temp = np.arange(1, Nin + 1)[t]
    return temp[np.cumsum(b) - 1]

# Convert index vector to vector of replication factors (vectorized method)
def repvector_vec(iv, Nin):
    return np.bincount(iv - 1, minlength=Nin)

##############################################################
#                        Main program                        #
##############################################################

N = 10
x = np.linspace(-3, 3, N)
w = stats.norm.pdf(x, 0, 1)
w = w / sum(w)

Nin = N;
Nout = 10
r = np.sort(x)

u = np.random.uniform()

r_sr = sr(w, Nin, Nout, u)  # Systematic resampling (SR)
r_rsr = rsr(w, Nin, Nout, u)  # Residual systematic resampling (RSR)
r_rr = rr(w, Nin, Nout, u)  # Residual resampling (RR)

r_rsr_mat = rsr_mat(w, Nin, Nout, u)  # Matrix version of RSR
r_rr_mat = rr_mat(w, Nin, Nout, u)  # Matrix version of RR

r_sr_vec = sr_vec(w, Nin, Nout, u)  # Vectorized version of SR
r_rsr_vec = rsr_vec(w, Nin, Nout, u)  # Vectorized version of RSR
r_rr_vec = rr_vec(w, Nin, Nout, u)  # Vectorized version of RR

iv = indexvector(r_sr, Nin)
iv_vec = indexvector_vec(r_sr, Nin)
rv_vec = repvector_vec(iv_vec, Nin)

r = r_sr

##############################################################
#                       Time functions                       #
##############################################################

f = ['sr', 'rsr', 'rr', 'rsr_mat', 'rr_mat', 'sr_vec', 'rsr_vec', 'rr_vec']
df = pd.DataFrame(data={'Function': f, 'Time': np.zeros(len(f))})
for i in np.arange(len(df)):
    t = process_time_ns()
    tmp = locals()[df.loc[i, 'Function']](w, Nin, Nout, u)
    df.loc[i, 'Time'] = (process_time_ns() - t) / 10**6
df['Normalized'] = df['Time'] / np.max(df['Time'])
df['Improvement'] = 1 / df['Normalized']
