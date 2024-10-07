import deepxde as xd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
#k=float(input("Spring constant"))
k=2
#A=float(input("Damping consatnt"))
A=3
#m=float(input("mass"))
m=2
#alpha=float(input("alpha"))
alpha=3
#P0=float(input("P0="))
P0=1
#P1=float(input("P1="))
P1=2
#Omega=float(input("Omega"))
Omega=3
#y0=float(input("y(0)"))
y0=2
#ydash0=float(input("y'(0)"))
ydash0=1
c=(A**2)/(alpha*m)
omega2=k/m

#Domain
geom=xd.geometry.TimeDomain(0, 1)
#diff eqn for data generation
#PINN
def ode(t,Y):
    x=Y[:,0:1]
    xdash=Y[:,1:2]
    x_der=xd.grad.jacobian(x, t)
    xdash_der=xd.grad.jacobian(xdash,t)
    return tf.concat([xdash-x_der,(xdash_der) + (c*xdash) + (x*omega2) - ((A*P0/m)+((A*P1*tf.sin(Omega*t))/m))],axis=1)
data=xd.data.TimePDE(geom,ode, [],3000,2,num_test=3000)
#defining Neural Netwrk
net=xd.nn.FNN([1]+8*[50]+[2],"tanh","Glorot normal")
#transforming input to periodic data
def input_tranform(t):
    return tf.concat([t,tf.sin(t),tf.sin(2*t),tf.sin(3*t),tf.sin(4*t),tf.sin(5*t),tf.sin(6*t)],axis=1)
#initial conditions
def hardconstraints(t,Y):
    r=Y[:,0:1]
    p=Y[:,1:2]
    return tf.concat([r*tf.tanh(t)+y0,p*tf.tanh(t)+ydash0],axis=1)
net.apply_feature_transform(input_tranform)
net.apply_output_transform(hardconstraints)
#model
model=xd.Model(data,net)
model.compile("adam",lr=0.001)
loss_history, train_state = model.train(iterations=5000)
model.compile("L-BFGS")
loss_history, train_state = model.train()
xd.saveplot(loss_history, train_state, issave=True, isplot=True)
#############################
#generating original data####
#############################
