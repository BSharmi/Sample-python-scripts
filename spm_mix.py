import numpy as np
import collections
from scipy import special as spec
import math


## spm_kl_normal1D function
def spm_kl_normal1D(m_q,c_q,m_p,c_p):
	Term1=0.5*(math.log(c_p))-0.5*(math.log(c_q));
	inv_c_p=1/c_p;
	Term2=0.5*(inv_c_p*c_q)+0.5*(m_q-m_p)*inv_c_p*(m_q-m_p);
	d=Term1+Term2-0.5;
	return d;

## spm_kl_gamma function
def spm_kl_gamma( b_q,c_q,b_p,c_p ):
	digamma_c_q=spec.psi(c_q);
	d=(c_q-1)*digamma_c_q-math.log(b_q)-c_q-spec.gammaln(c_q);
	d=d+spec.gammaln(c_p)+c_p*math.log(b_p)-(c_p-1)*(digamma_c_q+math.log(b_q));
	d=d+b_q*c_q/b_p;
	return d;

## spm_kmeans1 function
def spm_kmeans1( y, k ):
	#A 1-D array, containing the elements of the input, is returned.
	y=y.ravel('F')
	N=len(y)
	###Spread seeds evenly according to CDF
	x=sorted(y)
	i=sorted(range(len(y)), key=y.__getitem__)
	#pp=[1,2*np.ones(k-1)]
	pp= np.append([0],[2*np.ones(k-1)])
	seeds=map(lambda i: i * N/(2*k*1.0), pp)
	#seeds=[xx * N/(2*k*1.0) for xx in pp]
	import math
	seeds = np.asarray(map(math.ceil, np.cumsum(seeds))).astype(int)
	last_i=np.zeros(N);
	m=np.array(x)[seeds].astype('float');
	d=np.array([[0 for i in range(len(x))] for j in range(k)]);
	for loops in range(1,101):
		for j in range(0,k):
			d[j,:]=(y-m[j])**2			
		i=np.argmin(d,axis=0)
		if sum( np.subtract(i,last_i))==0:
			break
		else:
			for j in range(0,k):
				m[j]=np.mean(y[i==j])
			last_i=i;
	### Compute variances and mixing proportions
	v=np.zeros(k)
	pi=np.zeros(k)
	for j in range(0,k):
		v[j]=np.mean((y[i==j]-m[j])**2);
		pi[j]=len(y[i==j])/(N*1.0);
	# import collections
	# mix1=collections.namedtuple('mix1', 'v m pi k nloops assign');
	# mix1(v=v, m=m, pi=pi, k=k, nloops = loops,assign=i)
	mix1 = {‘v’: v, ‘m’: m, ‘pi’ : pi, ‘k’ : k, ‘nloops’ : loops, ‘assign’ : i}
	# return mix1(v=v, m=m, pi=pi, k=k, nloops = loops,assign=i);
	return mix1;


## spm_kl_dirichlet function
def spm_kl_dirichlet(*argv):
	if len(argv) <3:
		m=len(lambda_q);
		lambda_tot=sum(lambda_q);
		dglt=spec.psi(lambda_tot);
		log_tilde_pi=list();
		for s in range(m):
			log_tilde_pi.append(spec.psi(lambda_q[s])-dglt);
	else:
		lambda_q=argv[0];
		lambda_p=argv[1];
		log_tilde_pi=argv[2];
	d=spec.gammaln(sum(lambda_q));
	d=d+np.sum(np.multiply((lambda_q-lambda_p),log_tilde_pi));
	d=d-sum(spec.gammaln(lambda_q));
	d=d-spec.gammaln(np.sum(lambda_p));
	d=d+np.sum(spec.gammaln(lambda_p));
	return d;




y=np.array([[3,1,2,6,5,4]]).T
N,d= np.shape(y)
#spm_kmeans1.func_code.co_argcount
import collections
mix = collections.namedtuple('mix', 'nin m')
m=4 # number of components
mix(nin=d, m=m)
m_0=y.mean(axis=0)
lambda_0=1
beta_0=1
B_0=0.01*d*np.eye(d, dtype=int);
a_0=d
prior=collections.namedtuple('prior', 'lambda_0 m_0 beta_0 a_0 B_0');
prior=prior(lambda_0=lambda_0, m_0=m_0, beta_0=beta_0, a_0=a_0, B_0=B_0)
mix=collections.namedtuple('mix', 'nin m prior')
mix=mix(nin=d, m=m, prior=prior)
#from scipy import special as spec
sd=0.0
sd=sum(map(lambda xx: sd+ spec.psi((float)(a_0+1-1)/2),range(1,d+1)))
import math
log_tilde_gamma_0=sd-np.log(B_0)+math.log(2)

tmp= spm_kmeans1(y,m);
means=tmp.m;
covs=tmp.v;
priors=tmp.pi;
# Add pseudo-counts to ML priors
priors=map(lambda xx: xx + 1/(N*1.0), priors)
lambd = priors
state= [dict() for x in range(m)]
N_bar= [None] * m
for s in range(m):
	Cov=covs[s];
	state[s]={'m': means[s], 'beta': priors[s]*N+beta_0, 'a':priors[s]*N+a_0, 'B':priors[s]*N*Cov};
	N_bar[s]=priors[s]*N;

# Start algorithm
lik=[];
tol=0.001;
max_loops=100;
for loops in range(max_loops):
	# E-step
	lambda_tot=sum(lambd);
	log_tilde_pi=list();log_tilde_gamma=list();tilde_pi=list();tilde_gamma=list();
	gamma=np.array([[0 for i in range(N)] for j in range(m)],dtype=np.float);		
	for s in range(m):
		state[s].update({'bar_gamma':state[s].get('a')* (1/(state[s].get('B')))});
		log_tilde_pi.append(spec.psi(lambd[s])-spec.psi(lambda_tot));
		sd=0.0;
		sd=sum(map(lambda xx: sd+ spec.psi((state[s].get('a')+1-xx)/2),range(1,d+1)));
		log_tilde_gamma.append(sd-np.log(state[s].get('B'))+d*np.log(2));
		tilde_pi.append(np.exp(log_tilde_pi[s]));
		tilde_gamma.append(np.exp(log_tilde_gamma[s]));
		for n in range(N):
			gamma[s,n]=tilde_pi[s]*tilde_gamma[s]**0.5;
			dy=(y[n,0]-state[s].get('m'));
			gamma[s,n]=gamma[s,n]*(np.exp(-0.5*dy*state[s].get('bar_gamma')*dy)+np.finfo(float).eps)*np.exp(-d/(2*state[s].get('beta')));	
	
	gamma_n=np.sum(gamma,axis=0);
	for s in range(m):
		if np.mean(gamma_n) > np.finfo(float).eps:
			gamma[s,:]=gamma[s,:]/gamma_n;
	
	% M-step
	% Part -I
	pi_bar=list(); bar_mu=list();
	for s in range(m):
		pi_bar.append(np.mean(gamma[s,:])+np.finfo(float).eps);
		N_bar[s]=N*pi_bar[s]+np.finfo(float).eps;
		bar_mu.append(1/N_bar[s]*np.sum(np.multiply(np.matmul(gamma[s,:][np.newaxis].T,np.ones((1,d))),y),axis=0));

	 # get weighted means and covariances
	for s in range(m):
		state[s].update({'bar_sigma':np.zeros((d,1))});
		for n in range(N):
			dy=y[n]-bar_mu[s];
			state[s].update({'bar_sigma':state[s].get('bar_sigma')+np.multiply(gamma[s,n],(dy*dy))});
	
		state[s].update({'bar_sigma':(state[s].get('bar_sigma'))/N_bar[s]});	

	# now compute free energy
	f1=-spm_kl_dirichlet(lambd,lambda_0*np.ones((1,m)),log_tilde_pi);
	f2=list(); Cs=list(); C0=list(); f3=list(); f4=list(); f5=list(); fkl=list(); fkl_adj=list();
	for s in range(m):
		f2.append(np.asscalar(-spm_kl_gamma(state[s].get('a'),state[s].get(‘B’),a_0,B_0)));
		# KL-method for computing f3(s)
		Cs.append(state[s].get('B')/(state[s].get('beta')*state[s].get('a')));
		C0.append(state[s].get('B')/(beta_0*state[s].get('a')));
		f3.append(spm_kl_normal1D(state[s].get('m'),Cs[s],np.asscalar(m_0),C0[s]));
		# Check f3(s)
		check_f3=-0.5*d*(math.log(state[s].get('beta'))-math.log(beta_0)+(beta_0/state[s].get('beta'))-1);
		dm=np.asscalar(state[s].get(‘m’)-m_0);
		check_f3=check_f3-0.5*dm*beta_0*state[s].get('a')* (1/state[s].get('B')) *dm;
		f3[s]=check_f3;
		f4.append(N_bar[s]*log_tilde_pi[s]-sum(np.multiply(gamma[s,:],np.log(gamma[s,:]+np.finfo(float).eps))));
		LaB=0;LaB =spec.psi((state[s].get('a')+1-d)/2);
		LaB=LaB+d*math.log(2)-math.log((state[s].get(‘B’)));
		f5.append(np.asscalar(0.5*N_bar[s]*(-d*math.log(2*math.pi)+LaB-(state[s].get('bar_gamma')*state[s].get('bar_sigma'))-(d/state[s].get('beta')))));
		fkl.append(f2[s]+f3[s]+f4[s]+f5[s]);
		fkl_adj.append(f2[s]+f3[s]+(f4[s]+f5[s])/N_bar[s]);
	
	fm=f1+sum(f2)+sum(f3)+sum(f4)+sum(f5);
	acc=sum(f4)+sum(f5);
	kl_proportions=-f1;
    	kl_covs=-sum(f2);
    	kl_centres=-sum(f3);

	# Convergence criterion
	oldlik=lik;
    	lik=fm;
    	if loops>0:
        	if abs((lik-oldlik)/lik) < tol:
            	break;
	
	# Part-II
	for s in range(m):
		lambd[s]=N_bar[s]+lambda_0;
		state[s].update({'m':np.asscalar((N_bar[s]*bar_mu[s]+beta_0*m_0)/(N_bar[s]+beta_0))});
		state[s].update({'beta':N_bar[s]+beta_0});
		state[s].update({’a’:N_bar[s]+a_0});
		dy=np.asscalar(bar_mu[s]-m_0);
		state[s].update({'B':np.asscalar(N_bar[s]*state[s].get('bar_sigma') + (N_bar[s]*beta_0*dy*dy)/(N_bar[s]+beta_0)+B_0)});
