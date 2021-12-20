# Sinusoids method, and plot the channel output for fmT = 0.01, 0.1 
# and 0.5, and for M = 8, 16.
import numpy as np
import math
import scipy.special as sc
import matplotlib.pyplot as plt
fmT = [0.01,0.1,0.5]; 
T = 1; # simulation step size
fm = fmT; # fm
num = 3 ;
Omgp = 1; # Set average power as 1
sample_num = 300; # channel output data point
M = 8;
m = (np.arange(1, M+1));
N = 4*M+2;
n = (np.arange(1, N +1))
pi = 3.14159265
theta_n = 2*pi*n/N; 
theta_m = theta_n[0:M];
beta_m = np.tile(pi*m/M,(num,1));
alpha = 0;
# np.tile(np.cos(theta_m),(num,1));
d = np.tile(fm,(M,1));
e = np.transpose(d)
c =  np.tile(np.cos(theta_m),(num,1));
fn = e * c

cosfn = np.cos(2*pi*fm[0]*1*np.cos(theta_m));
cosfm = np.cos(2*pi*fm[0]*1);
gI = np.zeros((num,sample_num+1));
gQ = np.zeros((num,sample_num+1));

for t in range(0 , sample_num +1):
   x = np.cos(beta_m)*np.cos(2*pi*t*fn);
   xI = 2*x.sum(axis=1)
   yI=math.sqrt(2)*np.cos(np.multiply(2*pi, np.multiply(fm,t)))
   gI[:,t] = 2*(np.cos(beta_m)*np.cos(2*pi*t*fn)).sum(axis=1) + math.sqrt(2)*np.cos(alpha)*np.cos(np.multiply(2*pi, np.multiply(fm,t)))
   gQ[:,t] = 2*(np.sin(beta_m)*np.cos(2*pi*t*fn)).sum(axis=1) + math.sqrt(2)*np.sin(alpha)*np.cos(np.multiply(2*pi, np.multiply(fm,t)))
g = math.sqrt(2)*(gI + 1j*gQ)
#------------------#

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def absolute(x):
    y = np.sum(x) / np.size(x);
    return y

# a = np.abs(g[0])/np.abs(g[0]).mean(axis=0)
a = np.absolute(g)/mean2(np.absolute(g))


envelope_dB = 10*np.log(a);
########

# Plot the channel output
print (envelope_dB)
x_axis = np.arange(0,sample_num + 1);
plt.plot(x_axis,envelope_dB[0,:],'k',x_axis,envelope_dB[1,:],'b',x_axis,envelope_dB[2,:],'r' )
plt.title('Sum of Sinusoids Method for M=8');
plt.xlabel('Time, t/T');
plt.ylabel('Envelope Level (dB)');
plt.legend(['fmT=0.01','fmT=0.1','fmT=0.5']);
plt.grid();
plt.show();

plt.plot(x_axis,envelope_dB[0,:],'k')
plt.title('Sum of Sinusoids Method for M=8 (fmT=0.01)');
plt.xlabel('Time, t/T');
plt.ylabel('Envelope Level (dB)');
plt.grid();
plt.show();

plt.plot(x_axis,envelope_dB[1,:],'b')
plt.title('Sum of Sinusoids Method for M=8 (fmT=0.1)');
plt.xlabel('Time, t/T');
plt.ylabel('Envelope Level (dB)');
plt.grid();
plt.show();

plt.plot(x_axis,envelope_dB[2,:],'r' )
plt.title('Sum of Sinusoids Method for M=8 (fmT=0.5)');
plt.xlabel('Time, t/T');
plt.ylabel('Envelope Level (dB)');
plt.grid();
plt.show();


