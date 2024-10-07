import deepxde as xd
import tensorflow as tf
import math
import numpy as np

from matplotlib import pyplot as plt

def generate_data(t_min,t_max,domain_pts,bnd_pts):
    #Physical Constants

    #w_n=float(input("W_n:"))
    #zeta=float(input("Zeta:"))
    #Defining the differential Equation
    
    """def ode_fn(t, fn, additional_variables):
        mu, k, b = additional_variables
        with tf.GradientTape() as g_tt:
            g_tt.watch(t)
            with tf.GradientTape() as g_t:
                g_t.watch(t)
                x = fn(t)
            x_t = g_t.gradient(x, t)
        x_tt = g_tt.gradient(x_t, t)
        f = 1/k*x_tt + mu/k*x_t + x - b
    
        return f
    """
    #Initial Conditions
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
    denominator=((omega2-(Omega)**(2))**(2))+((c*Omega)**(2))
    def particular_solution(t):
        return ((A*P0)/(m*omega2))+(A*P1*(((omega2-(Omega)**2)*np.sin(Omega*t))-c*Omega*np.cos(Omega*t)))/(m*denominator)
    e=math.exp(1)
    def solution(t):
        if c**2<4*omega2:
            a=-c/2
            b=math.sqrt(4*omega2-c**2)/2
            c1=y0+((A*P1*Omega*c)/(m*(denominator)))-((A*P0)/(m*omega2))
            c2=(1/b)*((ydash0)-(c1*a)-((A*P1*Omega*(omega2-Omega**2))/(m*(denominator))))
            return (e**(a*t)*c1*np.cos(b*t))+(e**(a*t)*c2*np.sin(b*t))+(particular_solution(t))
        elif c**2==4*omega2:
            s=-c/2
            c1=y0-((A*P0)/(m*omega2))+((A*P1*c*Omega)/(m*denominator))
            c2=ydash0-c1*s+(A*P1*(Omega**2-omega2))/(m*denominator)
            return (c1*e**(s*t))+(c2*t*e**(s*t))+particular_solution(t)
        else:
            s1=(-c-np.sqrt(c**2-4*omega2))/2
            s2=(-c+np.sqrt(c**2-4*omega2))/2
            c1=1/(s1-s2)*(ydash0+((A*P0*s2)/(m*omega2))-(s2*y0)+((A*P1*(Omega*(Omega**2-omega2)-c*Omega*s2))/(m*denominator)))
            c2=y0-c1+((A*P1*c*Omega)/(m*denominator))-((A*P0)/(m*omega2))
            return (c1*e**(s1*t))+(c2*e**(s2*t))+particular_solution(t)

    def ode(x,y):
        y_der=xd.grad.jacobian(y,x,i=0)
        y_dder=xd.grad.hessian(y,x,i=0)
        return (y_dder) + (c*y_der) + (y*omega2) - ((A*P0/m)+((A*P1*tf.sin(Omega*x))/m))
    #y0=float(input("y(0)"))
    #ydash0=float(input("y'(0)"))
    geom=xd.geometry.TimeDomain(0,25)
    def boundary(x,on_initial):
        return xd.utils.isclose(x[0],0)and on_initial
    ic1=xd.icbc.IC(geom,lambda x:y0,boundary)
    def error(inputs,outputs,X):
        return xd.grad.jacobian(outputs,inputs,i=0)-ydash0
    ic2=xd.icbc.OperatorBC(geom,error, boundary)
    data=xd.data.TimePDE(geom,ode,[ic1,ic2],domain_pts,bnd_pts,solution=solution,num_test=500)
    #x=np.array(data.train_points())
    x=np.array(data.train_x)
    y=np.array(data.train_y)
    j=np.sort(x,axis=0)
    k=y[x.argsort(axis=0)]
    k_new=k.reshape((int(len(k)),1))
    print(k)
    print(j)
    filo=open("new_data.txt","w")
    for i in range(0,len(j)):
        filo.write(f"{float(j[i])}  {float(k_new[i])}\n",)
    together=np.array([j,k_new])
    filo.close()
    tog_new=np.reshape(together,(domain_pts+bnd_pts+2,2))
    plt.plot(j,k_new)
    file =open("deep_data.txt","w")
    np.savetxt(file,tog_new)
    return j,k_new
if __name__=="__main__":
    generate_data(0, 25, 3000, 2)