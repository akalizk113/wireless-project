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
M = 8;
m = (np.arange(1, M+1));

N = 4*M+2;
n = (np.arange(1, N+1));
pi = 3.14;
theta_n = 2*pi*n/N;  # theta_n is uniformly distributed
theta_m = theta_n[1:M+1];

print(theta_n)
print(theta_m)
# beta_m = np.tile(pi*m/M,(num,1));
# alpha = 0;
# fn = np.tile(fm,(1,M))*np.tile(np.cos(theta_m),(num,1));

# cosfn = np.cos(2*pi*fm(1)*1*np.cos(theta_m));
# cosfm = np.cos(2*pi*fm(1)*1);
# gI = np.zeros(num,sample_num+1);
# gQ = np.zeros(num,sample_num+1);

# for t in range(0 , sample_num) :
#    gI[:,t+1] = 2*sum(np.cos(beta_m)*np.cos(2*pi*t*fn),2)+math.sqrt(2)*np.cos(alpha)*np.cos(2*pi*fm*t);
#    gQ[:,t+1] = 2*sum(np.sin(beta_m)*np.cos(2*pi*t*fn),2)+math.sqrt(2)*np.sin(alpha)*np.cos(2*pi*fm*t);

# g = math.sqrt(2)*complex(gI, gQ)   

# tau = 10/fm; 
# phi1 = np.zeros(1, tau(1)+1);
# phi2 = np.zeros(1, tau(2)+1);
# phi3 = np.zeros(1, tau(3)+1);


# for i in range(0,tau(1)):
#    g_shift = np.zeros(1,sample_num+1)
#    g_shift[1, i+1:-1] = g[1,1:-1-i]
#    phi1[1,i+1] = np.mean(np.conj(g[1,:])*g_shift);

# for i in range(0,tau(2)):
#    g_shift = np.zeros(1,sample_num+1);
#    g_shift[1, i+1:-1] = g[2,1:-1-i];
#    phi2[1,i+1] = np.mean(np.conj(g[2,:])*g_shift);


# for i in range(0,tau(3)):
#    g_shift = np.zeros(1,sample_num+1);
#    g_shift[1, i+1:-1] = g[3,1:-1-i];
#    phi3[1,i+1] = np.mean(np.conj(g[3,:])*g_shift);




# k = 2*pi*fm(1)*(np.arange(0, tau(1)))*T

# S=sc.jv(0,k); # Ideal autocorrelation




# # Plot channel autocorrelation
# plt.plot(fm(1)*(np.arange(0, tau(1))), phi1/abs(phi1(1)),'r',fm(2)*(np.arange(0, tau(2))), phi2/abs(phi2(1)),'k',fm(3)*(np.arange(0, tau(3))), phi3/abs(phi3(1)),'b',fm(1)*(np.arange(0, tau(1))), S,'m--');
# plt.title('Autocorrelation of Sum of Sinusoids Method for M=8');
# plt.xlabel('f_m\tau');
# plt.ylabel('Autocorrelation');
# plt.legend('fmT=0.01','fmT=0.1','fmT=1','Ideal');

# plt.show()

###################################

# # For M = 16; basically the code below is the same as above
# M = 16;
# m = (1:M);
# N = 4*M+2;
# n = (1:N);
# theta_n = 2*pi*n/N;  # theta_n is uniformly distributed
# theta_m = theta_n(1:M);
# beta_m = repmat(pi*m/M,num,1);
# alpha = 0;
# fn = repmat(fm,1,M)*repmat(np.cos(theta_m),num,1);

# cosfn = np.cos(2*pi*fm(1)*1*np.cos(theta_m));
# cosfm = np.cos(2*pi*fm(1)*1);
# gI = np.zeros(num,sample_num+1);
# gQ = np.zeros(num,sample_num+1);

# # Use sum of sinusoids to derive gI and gQ
# for t = 0:sample_num
#     gI(:,t+1) = 2*sum(np.cos(beta_m)*np.cos(2*pi*t*fn),2)+math.sqrt(2)*np.cos(alpha)*np.cos(2*pi*fm*t);
#     gQ(:,t+1) = 2*sum(np.sin(beta_m)*np.cos(2*pi*t*fn),2)+math.sqrt(2)*np.sin(alpha)*np.cos(2*pi*fm*t);
# end
# g = math.sqrt(2)*(gI+1i*gQ);

# tau = 10/fm; # largest tau. Requirement: fm*tau=0~10
# # 3 autocorrelation with 3 different fm
# phi1 = np.zeros(1, tau(1)+1);
# phi2 = np.zeros(1, tau(2)+1);
# phi3 = np.zeros(1, tau(3)+1);

# # fm*T = 0.01
# for i=0:tau(1)
#     g_shift = np.zeros(1,sample_num+1);
#     g_shift(1, i+1:end) = g(1,1:end-i);
#     phi1(1,i+1) = mean(conj(g(1,:))*g_shift);
# end

# # fm*T = 0.1
# for i=0:tau(2)
#     g_shift = np.zeros(1,sample_num+1);
#     g_shift(1, i+1:end) = g(2,1:end-i);
#     phi2(1,i+1) = mean(conj(g(2,:))*g_shift);
# end

# # fm*T = 0.5
# for i=0:tau(3)
#     g_shift = np.zeros(1,sample_num+1);
#     g_shift(1, i+1:end) = g(3,1:end-i);
#     phi3(1,i+1) = mean(conj(g(3,:))*g_shift);
# end

# S=besselj(0,2*pi*fm(1)*(0:tau(1))*T); # Ideal autocorrelation

# # Plot channel autocorrelation
# figure,plot(fm(1)*(0:tau(1)), phi1/abs(phi1(1)),'r',fm(2)*(0:tau(2)), phi2/abs(phi2(1)),'k',fm(3)*(0:tau(3)), phi3/abs(phi3(1)),'b',fm(1)*(0:tau(1)), S,'m--');
# title('Autocorrelation of Sum of Sinusoids Method for M=16');
# xlabel('f_m\tau');
# ylabel('Autocorrelation');
# legend('fmT=0.01','fmT=0.1','fmT=1','Ideal');
# grid on