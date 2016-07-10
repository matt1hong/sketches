import math, numpy, random, json
import matplotlib.pyplot as plt
from scipy import stats
# numpy.random.seed(2)

def save_data(num_stimuli):
	for hurst in [0.2, 0.4, 0.6, 0.8]:
		# Experiment A
		with open('./data/H' + int(hurst*10) + '/exp-a/dataset.json', 'w') as outfile:
			dataset= []
			for corr in [0.9, -0.9]:
				for i in range(num_stimuli):
						bms = wfBm(H1= hurst, corr=corr)
						dataset.append(bms)
			json.dumps(dataset, outfile)

		# Experiment B # ADD DALC
		for corr in [0.9, -0.9]:
			for sensLabel in ['steep', 'shallow']:
				if corr > 0: 
					corrLabel = 'positive'
				elif corr < 0: 
					corrLabel = 'negative'
				if sensLabel = 'steep':
					slopeCoeffs = [2,4]
				else:
					slopeCoeffs = [0,2]
				with open('./data/H' + int(hurst*10) + '/exp-b/' + corrLabel + sensLabel + '/dataset.json', 'w') as outfile:
					dataset= []
					for i in range(num_stimuli):
						bms = wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs)
						dataset.append(bms)
					json.dumps(dataset, outfile)

	pass

def wfBm(N = 100, H1=8./10., corr=0.9, study=True, minDiff = 0, slopeCoeffs = [0,4], DALC = False):
	dataset = {}

	H2 = H1
	HH = [2*H1, 2*H2]
	covariance = numpy.zeros((N,N))

	# initialize cov matrix
	A = [[],[]]
	for k in range(2):
		A[k] = numpy.zeros((N,N))
		for i in range(N):
		    for j in range(N):
		        d = abs(i-j)
		        # cov is dependent on Hurst
		        covariance[i,j] = (abs(d - 1)**HH[k] + (d + 1)**HH[k] - 2*d**HH[k])/2.
		# eigenvalues of the cov matrix
		w,v = numpy.linalg.eig(covariance)
		# A[k] is the sqrt of cov matrix
		for i in range(N):
		    for j in range(N):
		        A[k][i,j] = sum(math.sqrt(w[k])*v[i,k]*v[j,k] for k in range(N))
		
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
			eta = numpy.dot(A[i],x[i])
			xfBm.append([sum(eta[0:i]) for i in range(len(eta)+1)])
		# xBM = [sum(x[0:i]) for i in range(len(x)+1)]
		bm = numpy.dot(R, xfBm)
		r, p = stats.pearsonr(bm[0],bm[1])

		# check criteria and return if met
		if abs(float(r) - corr) < 0.05:
			# normalize
			for i in range(2):
				minV = min(bm[i])
				if minV < 0:
					bm[i] = [v - minV for v in bm[i]]

			maxV = max([inner for outer in bm for inner in outer])
			bm[0] = [v/maxV for v in bm[0]]
			bm[1] = [v/maxV for v in bm[1]]

			# check conditions, randomize start 
			rangeArrs = []
			if study:
				for i in range(2):
					if min(bm[i]) == 0: 
						extV = max(bm[i])
						rangeArrs[i] = [0, extV]

						if extV < 0.2:
							break
						# shift up by random number not exceeding 1-extV
						randV = random.random() * (1-extV)
						bm[i] = [n + randV for n in bm[i]]
					else: 
						# range is [extV, 1]
						extV = min(bm[i])
						rangeArrs[i] = [extV, 1]
						if extV > 0.8 or abs(rangeArrs[0][1] - (1-extV)) < minDiff:
							# second condition: difference in range is at least minDiff
							break
						# shift down by random number not exceeding extV
						randV = random.random() * extV
						bm[i] = [n - randV for n in bm[i]]
				else:
					# minimum conditions met
					continue
				break # not met

			# regression
			slope, intercept, r_value, p_value, std_err = stats.linregress(bm[0],bm[1])

			# pi/16 = 11.25, pi/8 = 22.5
			if abs(slope) >= math.tan(math.pi/8)*minSlopeCoeff and abs(slope) <= math.tan(math.pi/8)*maxSlopeCoeff:
				dataset['Data'] = bm
				dataset['Blue range'] = rangeArr[0]
				dataset['Green range'] = rangeArr[1]
				dataset['Regression slope'] = slope
				dataset['Correlation'] = float(r)
				dataset['Sign of correlation'] = numpy.sign(r)
				if slope > 1:
					dataset['Steepness'] = 'Steep'
				else:
					dataset['Steepness'] = 'Shallow'
				
				return dataset

			# if slope >= math.tan(math.pi/8)*0 and slope <= math.tan(math.pi/8)*1:
			# 	dataset['slopeCat'] = 1

def show(type = 'CS'):
	rhos = []
	for i in range(1000):
		bms = wfBm(corr=0.9)
		if type == "DALC":
			plt.plot(bms[0])
			plt.plot(bms[1])
		else:
			ax = plt.gca()
			ax.set_autoscale_on(False)
			ax.set_aspect('equal')
			plt.plot(bms[0],bms[1])
		plt.show()
		return
	# 	rhos.append(corr)
	# plt.hist(rhos)
	# plt.show()

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