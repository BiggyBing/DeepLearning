import numpy as np

import matplotlib.pyplot as plt

 

# Get x values of the cosine wave

#time        = np.arange(0,20,0.1);
#
#sample_p = np.arange(0,20,0.25) 

# Amplitude of the cosine wave is cosine of a variable like time


# Plot a cosine wave using time and amplitude obtained for the cosine wave

#plot.subplot(221)

x = np.arange(0,12)
_16PSK, = plt.plot(x, test_feature_1[9055])
BPSK, = plt.plot(x, test_feature_1[7424], '--')

plt.legend(handles = [_16PSK, BPSK], labels = ['16PSK_001101011111','QPSK_110100xxxxxx'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

#plt.title('Comparison between BPSK and QPSK')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which='both')
plt.axhline(y=0, color='b')
#plot.subplot(222)
#plot.title('QPSK wave')
#plot.xlabel('Time')
#plot.ylabel('Amplitude = cosine(time)')
#plot.grid(True, which='both')
#plot.axhline(y=0, color='b')
#plot.plot(np.arange(0,3*g_Tc,g_Tc/50), QPSK[0][0:150])
#
#plot.subplot(223)
#plot.title('16PSK wave')
#plot.xlabel('Time')
#plot.ylabel('Amplitude = cosine(time)')
#plot.grid(True, which='both')
#plot.axhline(y=0, color='b')
#plot.plot(np.arange(0,5*g_Tc,g_Tc/50), _16PSK[0][0:250])
#

plot.show()