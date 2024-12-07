import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
DTYPE = "float32"
tf.keras.backend.set_floatx(DTYPE)


def load_data(d1=0, d2=25):
    # loading generated data
    XDE_data = np.loadtxt("new_data.txt")
    t_data = XDE_data[:,0]
    x_data = XDE_data[:,1]
    plt.plot(t_data,x_data)
    # collocation points for enforcing ODE (whole window of interest)
    t_physics = XDE_data[:,0]

    # convert arrays to tf tensors
    t_data_tf = tf.convert_to_tensor(t_data, dtype=DTYPE)
    x_data_tf = tf.convert_to_tensor(x_data, dtype=DTYPE)
    #x_data_un_tf = tf.convert_to_tensor(XDE_data[d1:d2,1]/1e06, dtype=DTYPE)
    t_physics_tf = tf.convert_to_tensor(t_physics, dtype=DTYPE)

    T_data = tf.reshape(t_data_tf[:], shape=(t_data.shape[0], 1))
    X_data = tf.reshape(x_data_tf[:], shape=(x_data.shape[0], 1))
    
    #X_data_un = tf.reshape(x_data_un_tf[:], shape=(x_data.shape[0], 1))
    
    T_r = tf.reshape(t_physics_tf[:], shape=(t_physics.shape[0], 1))

    # pick the exact (smoothed) data
    T_exact = XDE_data[:,0].copy()
    X_exact =XDE_data[:,1].copy()

    return T_data, X_data, T_r, T_exact, X_exact#, (days[d1:d2]-d1)/365, XDE_data[d1:d2,1]/1e06, X_data_un



