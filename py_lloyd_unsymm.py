# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:43:29 2018

@author: Vijay
"""
from scipy import integrate
from scipy.optimize import fsolve
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
alp = 4
bet = 2
y = beta.pdf(x,alp,bet)
k = 10
#q = np.sort(np.random.rand(1,k))
qinit = np.linspace(0,1,num = k,endpoint = True).reshape(1,k)
q = np.zeros((1,k))
q_true = np.zeros((1,k))
for b in range(k):
    q[0,b] = qinit[0,b]
    q_true[0,b] = qinit[0,b]
fig1 = plt.figure(figsize=(5,5))
plt.plot(x,y,color = 'k')
#plt.scatter(q,-0.1*np.ones((1,k)), color = 'green')
plt.ion()
q[0,0] = 0
q[0,k-1] = 1
#
def myIntFunc(v,l):
    return 2*(l-v)*beta.pdf(v,alp,bet)
#
def myIntFunc2(v,l,xans):
    return 2*(l-v)*(xans[0]*v**2+xans[1]*v+xans[2])
#
def myEqn(l,xpp1,xpm1,m,c):
    return 1.0/8*(l-xpp1)**2*(m*(l+xpp1)/2+c) - 1.0/8*(l-xpm1)**2*(m*(l+xpm1)/2+c) + 2*(l*c/2*(xpp1-xpm1)+1.0/8*(l*m-c)*((l+xpp1)**2-(l+xpm1)**2)-m/24*((l+xpp1)**3-(l+xpm1)**3))
#    
def myFunc(l,xpm1,xpp1):
    y = 1.0/8*(l-xpp1)**2*beta.pdf(l/2+xpp1/2,alp,bet) - 1.0/8*(l-xpm1)**2*beta.pdf(l/2+xpm1/2,alp,bet) +integrate.quad(myIntFunc,l/2+xpm1/2,l/2+xpp1/2,args=(l,))
    #integrate.quad(myIntFunc,xpm1,l,args=(l,)) - (xpp1-l)**2*(beta.pdf(l,alp,bet))
    return y
#
def poly_calc(x,xans):
    return (xans[0]*x**2+xans[1]*x+xans[2])
#
def myFunc2(l,xpm1,xpp1,xans):
    y = 1.0/8*(l-xpp1)**2*poly_calc(l/2+xpp1/2,xans) - 1.0/8*(l-xpm1)**2*poly_calc(l/2+xpm1/2,xans) +integrate.quad(myIntFunc2,l/2+xpm1/2,l/2+xpp1/2,args=(l,xans))
    return y
#
itr = 1
while itr <20:
    '''plt.clf()
    plt.plot(x,y)
    plt.scatter(qinit,0*np.ones((1,k)), color = 'green')
    plt.scatter(q_true,-0.1*np.ones((1,k)), color = 'orange', marker = 'D')
    plt.scatter(q,np.zeros((1,k)), color = 'orange')
    #plt.title('Iteration # = ' + str(itr))
    plt.grid()
    plt.pause(0.5)
    #plt.show()
    #time.sleep(1)'''
    for a in range(2,k-1,2): # Loop for even points
        xpp1 = q[0,a+1]
        xpm1 = q[0,a-1]
        ypp1 = beta.pdf(xpp1,alp,bet)
        ypm1 = beta.pdf(xpm1,alp,bet)
        #
        #myFunc = @(l) quad(@(v)2*(l-v).*(-6*v.*(v-1)),xpm1,l) - (xpp1-l)^2*(-6*l.*(l-1));
        q_true[0,a] = fsolve(myFunc,args=(xpm1,xpp1),x0 = xpp1/2+xpm1/2);
        #
        if (xpm1 == 0) or (xpp1 == 1):
            xp = q[0,a]
            yp = beta.pdf(xp,alp,bet)
            e = np.array([[xpm1**2, xpm1, 1],[xp**2,xp,1],[xpp1**2,xpp1,1]])
            f = np.array([ypm1,yp,ypp1])
            xans = np.linalg.solve(e, f)
            q[0,a] = fsolve(myFunc2,args=(xpm1,xpp1,xans),x0 = xpp1/2+xpm1/2)
        else:
            m = (ypp1-ypm1)/(xpp1 - xpm1)
            c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)
            q[0,a] = fsolve(myEqn,args=(xpp1,xpm1,m,c),x0 = xpp1/2+xpm1/2)
    #
    for a in range(1,k-1,2): # Loop for odd points
        xpp1 = q[0,a+1]
        xpm1 = q[0,a-1]
        ypp1 = beta.pdf(xpp1,alp,bet)
        ypm1 = beta.pdf(xpm1,alp,bet)
        #
        #myFunc = @(l) quad(@(v)2*(l-v).*(-6*v.*(v-1)),xpm1,l) - (xpp1-l)^2*(-6*l.*(l-1));
        q_true[0,a] = fsolve(myFunc,args=(xpm1,xpp1),x0 = xpp1/2+xpm1/2);
        #
        if (xpm1 == 0) or (xpp1 == 1):
            xp = q[0,a]
            yp = beta.pdf(xp,alp,bet)
            e = np.array([[xpm1**2, xpm1, 1],[xp**2,xp,1],[xpp1**2,xpp1,1]])
            f = np.array([ypm1,yp,ypp1])
            xans = np.linalg.solve(e, f)
            q[0,a] = fsolve(myFunc2,args=(xpm1,xpp1,xans),x0 = xpp1/2+xpm1/2)
        else:
            m = (ypp1-ypm1)/(xpp1 - xpm1)
            c = (ypm1*xpp1 - ypp1*xpm1)/(xpp1-xpm1)
            q[0,a] = fsolve(myEqn,args=(xpp1,xpm1,m,c),x0 = xpp1/2+xpm1/2)
    itr += 1

# Compute the error encountered
def errCostFunc(x,l):
    return (l-x)**2*beta.pdf(x,alp,bet)


err = 0
for aa in range(1,k-1):
    if k==3:
        err += integrate.quad(errCostFunc,q[0,aa-1],q[0,aa+1],args=(q[0,aa],))[0]
    elif aa == 1:
        err += integrate.quad(errCostFunc,q[0,aa],0.5*q[0,aa+1]+0.5*q[0,aa],args=(q[0,aa],))[0]
    elif aa==k-1:
        err += integrate.quad(errCostFunc,0.5*q[0,aa-1]+0.5*q[0,aa],q[0,aa+1],args=(q[0,aa],))[0]
    else:
        err += integrate.quad(errCostFunc,0.5*q[0,aa-1]+0.5*q[0,aa],0.5*q[0,aa+1]+0.5*q[0,aa],args=(q[0,aa],))[0]

print('Total Approx. Error is '+ str(err))


err_true = 0
for aa in range(1,k-1):
    if k==3:
        err_true += integrate.quad(errCostFunc,q_true[0,aa-1],q_true[0,aa+1],args=(q_true[0,aa],))[0]
    elif aa == 1:
        err_true += integrate.quad(errCostFunc,q_true[0,aa],0.5*q_true[0,aa+1]+0.5*q_true[0,aa],args=(q_true[0,aa],))[0]
    elif aa==k-1:
        err_true += integrate.quad(errCostFunc,0.5*q_true[0,aa-1]+0.5*q_true[0,aa],q_true[0,aa+1],args=(q_true[0,aa],))[0]
    else:
        err_true += integrate.quad(errCostFunc,0.5*q_true[0,aa-1]+0.5*q_true[0,aa],0.5*q_true[0,aa+1]+0.5*q_true[0,aa],args=(q_true[0,aa],))[0]

print('Total True Error is '+ str(err_true))

plt.scatter(q,np.zeros((1,k)), color = 'black',facecolors='none', label='Linear Approx.')
plt.scatter(q_true,-0.1*np.ones((1,k)), color = 'black', marker = 'D', facecolors='none',label='True Solution')
#plt.scatter(q_init,-0.1*np.ones((1,k)), color = 'green')
plt.grid(alpha=0.45)
plt.xlabel('$x$',fontsize=18)
plt.ylabel('$f_X(x)$',fontsize=18)
plt.legend(bbox_to_anchor=(0.875,0.3),loc =5,fontsize=18, framealpha=0.95)
#plt.legend(fontsize=18)
plt.savefig('fig_lloyd_app_true.pdf',bbox_inches='tight')
plt.show()

