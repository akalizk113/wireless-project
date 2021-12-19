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
sample_num = 30000; # channel output data point
M = 16;
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


tau = np.divide(10, fm)

# print(tau)
phi1 = np.zeros((1, int(tau[0]+1)), dtype = 'complex_');
phi2 = np.zeros((1, int(tau[1]+1)), dtype = 'complex_');
phi3 = np.zeros((1, int(tau[2]+1)), dtype = 'complex_');

# for i in range(0, int(tau[0] +1)):
for i in range(0, int(tau[0] +1)):
   g_shift = np.zeros((1,sample_num+1), dtype = 'complex_')
   if(i == 0): 
      g_shift[0][i:] = g[0]
      phi1[0,i] = np.multiply(g[0].conj(), g_shift[0]).mean(axis=0); 
   else :
      g_shift[0][i:] = g[0][:-i]
      phi1[0,i] = np.multiply(g[0].conj(), g_shift[0]).mean(axis=0);
   



for i in range(0, int(tau[1] +1)):
   g_shift = np.zeros((1,sample_num+1), dtype = 'complex_')
   if(i == 0): 
      g_shift[0][i:] = g[1]
      phi2[0,i] = np.multiply(g[1].conj(), g_shift[0]).mean(axis=0);
      continue
   else :
      g_shift[0][i:] = g[1][:-i]
      phi2[0,i] = np.multiply(g[1].conj(), g_shift[0]).mean(axis=0);


for i in range(0, int(tau[2] +1)):
   g_shift = np.zeros((1,sample_num+1), dtype = 'complex_')
   if(i == 0): 
      g_shift[0][i:] = g[2]
      phi3[0,i] = np.multiply(g[2].conj(), g_shift[0]).mean(axis=0);
      continue
   else :
      g_shift[0][i:] = g[2][:-i]
      phi3[0,i] = np.multiply(g[2].conj(), g_shift[0]).mean(axis=0);



k = 2*pi*fm[0]*(np.arange(0, tau[0] +1))*T

S=sc.jv(0,k); # Ideal autocorrelation

# Plot channel autocorrelation



plt.plot(np.multiply(fm[0], np.arange(0, tau[0] + 1)), phi1[0]/np.abs(phi1[0][0]), 'r',fm[1]*(np.arange(0, tau[1] + 1)), phi2[0]/abs(phi2[0][0]), 'k',fm[2]*(np.arange(0, tau[2] + 1)), phi3[0]/abs(phi3[0][0]),'b',fm[0]*(np.arange(0, tau[0] + 1)), S,'m--');

plt.title('Autocorrelation of Sum of Sinusoids Method for M=16');
plt.xlabel('f_m\tau');
plt.ylabel('Autocorrelation');
plt.legend(['fmT=0.01','fmT=0.1','fmT=1','Ideal']);

plt.grid()
plt.show()


# print(phi1/abs(phi1[0]))