# coding: utf-8

import numpy as np
import pymc as pm
#K, V, D = 2, 4, 3 # number of topics, words, documents
K, V, D = 5, 10, 20 # number of topics, words, documents

#data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

data = np.random.randint(0,10,size=(D,V))



#data = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])
alpha = np.ones(K)
beta = np.ones(V)

theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i, pm.Dirichlet("ptheta_%s" % i, theta=alpha)) for i in range(D)])
phi = pm.Container([pm.CompletedDirichlet("phi_%s" % i, pm.Dirichlet("pphi_%s" % i, theta=beta)) for i in range(K)])
Wd = [len(doc) for doc in data]

z = pm.Container([pm.Categorical("z_%s" % d, p=theta[d], size=Wd[d], value=np.random.randint(K,size=Wd[d])) for d in range(D)])
w = pm.Container([pm.Categorical("w_%s,%s" % (d,i), p=pm.Lambda("phi_z_%s_%s" % (d,i), lambda z=z[d][i], phi=phi: phi[z]), value=data[d][i], observed=True) for d in range(D) for i in range(Wd[d])])

model = pm.Model([theta, phi, z, w])
mcmc = pm.MCMC(model)

# Fit the model by sampling from the data 1000 times with burn in of 10.
mcmc.sample(1000, burn=10)

# Show distribution of topics over words
print phi.value

# Show distribution of words in documents over topics
print w.value
