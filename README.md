This Repository is based on the repository of LivingMatterLab [here](https://github.com/LivingMatterLab/xPINNs/tree/main).
To reproduce the results on Windows Machine:
## Steps:
* Install Anaconda
* Use the file "Windows_requirements.txt" to setup a conda environment. Instructions on how to create the environment file is in the file.
* After settng up the environment, First run the file 'generate_data.py'. The file generates data points and writes it to "deep_data.txt".
* Now, restart the console or use a new console from this point onwards. This step is essential since generate_data.py uses deepXDE which sets the flag tf.enable_eager_execution to false which is true by default and is required to be true for the remaining execution.
* Now, run the file BPINN_HMC.py.The program will generate a graph comparnng the neural network results with the true value.
