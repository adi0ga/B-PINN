import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
DTYPE = "float32"
tf.keras.backend.set_floatx(DTYPE)


def load_data(d1=350, d2=700):
    # world-wide COVID cases, daily, since start of pandemic
    covid_world = np.loadtxt("new_data.txt")
    t_data = covid_world[:,0]
    x_data = covid_world[:,1]
    plt.plot(t_data,x_data)
    # collocation points for enforcing ODE (whole window of interest)
    t_physics = covid_world[:,0]

    # convert arrays to tf tensors
    t_data_tf = tf.convert_to_tensor(t_data, dtype=DTYPE)
    x_data_tf = tf.convert_to_tensor(x_data, dtype=DTYPE)
    #x_data_un_tf = tf.convert_to_tensor(covid_world[d1:d2,1]/1e06, dtype=DTYPE)
    t_physics_tf = tf.convert_to_tensor(t_physics, dtype=DTYPE)

    T_data = tf.reshape(t_data_tf[:], shape=(t_data.shape[0], 1))
    X_data = tf.reshape(x_data_tf[:], shape=(x_data.shape[0], 1))
    
    #X_data_un = tf.reshape(x_data_un_tf[:], shape=(x_covid.shape[0], 1))
    
    T_r = tf.reshape(t_physics_tf[:], shape=(t_physics.shape[0], 1))

    # pick the exact (smoothed) data
    T_exact = covid_world[:,0].copy()
    X_exact =covid_world[:,1].copy()

    return T_data, X_data, T_r, T_exact, X_exact#, (days[d1:d2]-d1)/365, covid_world[d1:d2,1]/1e06, X_data_un



