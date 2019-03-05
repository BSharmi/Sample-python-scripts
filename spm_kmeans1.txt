def spm_kmeans1( y, k ):
	y=y.ravel('F')
	N=len(y)
	###Spread seeds evenly according to CDF
	x=sorted(y)
	i=sorted(range(len(y)), key=y.__getitem__)
	#pp=[1,2*np.ones(k-1)]
	pp= np.append([0],[2*np.ones(k-1)])
	seeds=[xx * N/(2*k*1.0) for xx in pp]
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
	
	## using named tuples
	import collections
	mix1=collections.namedtuple(‘mix1’, ‘v m pi k nloops assign’);
	mix1(v=v, m=m, pi=pi, k=k, nloops = loops,assign=i)
	## using dictionary by dict function
	mix1=dict([(‘v’, v),(‘m’,m),(‘pi’,pi),(‘k’,k),(‘nloops’,loops),(‘assign’,i)])
	## or normal dictionary function
	mix1 = {‘v’: v, ‘m’: m, ‘pi’ : pi, ‘k’ : k, ‘nloops’ : loops, ‘assign’ : i}
	return mix1(v=v, m=m, pi=pi, k=k, nloops = loops,assign=i);


