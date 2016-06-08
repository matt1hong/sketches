import math, numpy
import matplotlib.pyplot as plt
import scipy.stats as stats
# numpy.random.seed(2)

def wfBm(N = 40, H=2./10.,corr=-0.2):
	HH = 2*H
	covariance = numpy.zeros((N,N))

	# initialize cov matrix
	A = numpy.zeros((N,N))
	for i in range(N):
	    for j in range(N):
	        d = abs(i-j)
	        # cov is dependent on Hurst
	        covariance[i,j] = (abs(d - 1)**HH + (d + 1)**HH - 2*d**HH)/2.
	# eigenvalues of the cov matrix
	w,v = numpy.linalg.eig(covariance)
	# A is the sqrt of cov matrix
	for i in range(N):
	    for j in range(N):
	        A[i,j] = sum(math.sqrt(w[k])*v[i,k]*v[j,k] for k in range(N))
	
	while True:
		# Generate two random series
		x = numpy.random.randn(2,N)
		# Cholesky factorization
		R = numpy.linalg.cholesky([[1.0, corr],[corr, 1.0]])
		# Generate correlated series
		# path = numpy.dot(R,x)

		# Correlate the noise
		xfBm = []
		for i in range(2):
			eta = numpy.dot(A,x[i])
			xfBm.append([sum(eta[0:i]) for i in range(len(eta)+1)])
		# xBM = [sum(x[0:i]) for i in range(len(x)+1)]
		bm = numpy.dot(R, xfBm)
		r, p = stats.pearsonr(bm[0],bm[1])
		return bm
		# if float(abs(r)) >= abs(corr) - 0.05 and float(abs(r)) <= abs(corr) + 0.05:
		# 	return bm

def show():
	rhos = []
	for i in range(1000):
		bms = wfBm(20)
		corr, p = stats.pearsonr(bms[0],bms[1])
		rhos.append(corr)
	plt.hist(rhos)
	plt.show()

# corr = 0
# i = 0
# # while True:
# # 	print(abs(max(values1) - min(values1)))
# # 	values1 = wfBm()
# while corr < 0.8:
# 	i += 1
# 	print(i)
# 	values1 = wfBm()
# 	values2 = wfBm()
# 	corr, p = stats.pearsonr(values1, values2)

# plt.title('fBm (blue) vs BM (red)')
# plt.xlabel('i')
# plt.ylabel('x(i)')
# plt.plot(values1,'b.-')
# plt.plot(values2,'r.-')
# plt.show()