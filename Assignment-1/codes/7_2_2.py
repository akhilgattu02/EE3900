
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

input_signal,fs = sf.read('filter_codes_Sound_Noise.wav') 
sampl_freq=fs
order=4   
cutoff_freq=4000.0

Wn=2*cutoff_freq/sampl_freq  

b, a = signal.butter(order, Wn, 'low') 


def H(z):
	num = np.polyval(b,z**(-1))
	den = np.polyval(a,z**(-1))
	H = num/den
	return H
		

omega = np.linspace(0,np.pi,100)


plt.plot(omega, abs(H(np.exp(1j*omega))))
plt.xlabel('$\omega$')
plt.ylabel('$|H(e^{\jmath\omega})| $')
plt.grid()


plt.show()