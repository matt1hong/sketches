import math, numpy, random, json
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import stats
# numpy.random.seed(2)

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def save_data(num_stimuli):
	for hurst in [0.2, 0.4, 0.6, 0.8]:
	# 	# Experiment A
	# 	with open('data/H' + str(int(hurst*10)) + '-a.json', 'w') as outfile:
	# 		datasetA = []
	# 		for corr in [0.9, -0.9]:
	# 			for i in range(num_stimuli):
	# 				print(i)
	# 				bms = wfBm(H1= hurst, corr=corr)
	# 				datasetA.append(bms)
	# 		json.dump(dataset, outfile)
	# 	print('A done')

		# Experiment B # ADD DALC
		# for corr in [0.9, -0.9]:
		# 	corrLabel = 'positive' if corr > 0 else 'negative'
		# 	with open('data/H' + str(int(hurst*10)) + '-b-' + corrLabel + '.json', 'w') as outfile:
		# 		dataset= []
		# 		for slopeCoeffs in [[2,4],[0,2]]:
		# 			for i in range(num_stimuli):
		# 				print(i)
		# 				bms = wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs)
		# 				dataset.append(bms)
		# 		json.dump(dataset, outfile)
		# print('B done')

		# Experiment C # ADD DALC
		for corr in [0.9, -0.9]:
			for sensLabel in ['steep', 'shallow']:
				i = 0
				corrLabel = 'positive' if corr > 0 else 'negative'
				with open('data/H' + str(int(hurst*10)) + '-c-' + corrLabel + '-' + sensLabel + '.json', 'w') as outfile:
					dataset=[]
					while i < num_stimuli:
						slopeCoeffs = [2,4] if sensLabel == 'steep' else [0,2]
						bms1 = wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs)
						bms2 = wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs)

						print('hi')
						# scale = ((random.random()-0.5)*0.6+0.5)

						# bms1['values1'] = [n * 0.6 for n in bms1['values1']]

						# ax = plt.gca()
						# ax.set_autoscale_on(False)
						# ax.set_aspect('equal')
						# plt.plot(bms1['values1'],bms1['values2'])
						# plt.show()
						# ax = plt.gca()
						# ax.set_autoscale_on(False)
						# ax.set_aspect('equal')
						# plt.plot(bms2['values1'],bms2['values2'])
						# plt.show()
						print(abs(bms1['Regression slope']))
						print(abs(bms2['Regression slope']))
						plt.plot(bms1['values1'])
						plt.plot(bms1['values2'])
						plt.show()
						plt.plot(bms2['values1'])
						plt.plot(bms2['values2'])
						plt.show()
						# if abs(abs(bms1['Blue range value'] - bms1['Green range value']) - 
						# 			abs(bms2['Blue range value'] - bms2['Green range value'])) > 0.2:
						if 
							# Diff at the highest level of interaction
							# ax = plt.gca()
							# ax.set_autoscale_on(False)
							# ax.set_aspect('equal')
							# plt.plot(bms1['values1'],bms1['values2'])
							# plt.show()
							# ax = plt.gca()
							# ax.set_autoscale_on(False)
							# ax.set_aspect('equal')
							# plt.plot(bms2['values1'],bms2['values2'])
							# plt.show()

							plt.plot(bms1['values1'])
							plt.plot(bms1['values2'])
							plt.show()
							plt.plot(bms2['values1'])
							plt.plot(bms2['values2'])
							plt.show()
							print(i)
							i += 1
							dataset.append([bms1, bms2])
					json.dump(dataset, outfile)
		print('C done')

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
	
	over_lim = 0
	count = 0
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
			count += 1
			print(count)
			# normalize
			for i in range(2):
				minV = min(bm[i])
				if minV < 0:
					bm[i] = [v - minV for v in bm[i]]

			lim = 10
			maxV = max([inner for outer in bm for inner in outer])
			if maxV > lim: 
				over_lim += 1
				print(str(over_lim) + " over 10")
				continue
			bm[0] = [v/lim for v in bm[0]]
			bm[1] = [v/lim for v in bm[1]]

			# check conditions, randomize start 
			rangeArrs = []
			if study:
				should_continue = False
				for i in range(2):
					if min(bm[i]) == 0: 
						extV = max(bm[i])
						if extV < minDiff:
							should_continue = True
							break

						# shift up by random number not exceeding 1-extV
						randV = random.random() * (1-extV)
						bm[i] = [n + randV for n in bm[i]]
					else: 
						# range is [extV, 1]
						extV = min(bm[i])
						if extV > 1-minDiff:
							should_continue = True
							break

						# shift down by random number not exceeding extV
						randV = random.random() * extV
						bm[i] = [n - randV for n in bm[i]]

					if i == 0: 
						scale = (random.random()*0.8+0.2)
						bm[0] = [n * scale for n in bm[0]]
					rangeArrs.append([min(bm[i]), max(bm[i])])

				if should_continue \
					or abs(numpy.diff(rangeArrs[0]) - numpy.diff(rangeArrs[1])) < minDiff: 
					continue # generate new series 

			# regression
			slope, intercept, r_value, p_value, std_err = stats.linregress(bm[0],bm[1])

			# pi/16 = 11.25, pi/8 = 22.5
			# scale = ((random.random()-0.5)*0.6+0.5)
			# bm[0] = [n * 0.6 for n in bm[0]]
			if abs(slope) >= math.tan(slopeCoeffs[0]*math.pi/8) and abs(slope) <= math.tan(slopeCoeffs[1]*math.pi/8):
				print(slope)
				dataset['values1'] = bm[0].tolist()
				dataset['values2'] = bm[1].tolist()
				dataset['Blue range'] = rangeArrs[0]
				dataset['Blue range value'] = rangeArrs[0][1] - rangeArrs[0][0]
				dataset['Green range'] = rangeArrs[1]
				dataset['Green range value'] = rangeArrs[1][1] - rangeArrs[1][0] 
				dataset['Regression slope'] = slope
				dataset['Correlation'] = float(r)
				dataset['Sign of correlation'] = numpy.sign(r)
				if slope > 1:
					dataset['Steepness'] = 'Steep'
				else:
					dataset['Steepness'] = 'Shallow'
				
				return dataset
			else:
				continue

			# if slope >= math.tan(math.pi/8)*0 and slope <= math.tan(math.pi/8)*1:
			# 	dataset['slopeCat'] = 1

def show(type = 'CS'):
	rhos = []
	for i in range(1000):
		hurst = 0.2
		corr = 0.9
		minDiff = 0.2
		slopeCoeffs = [2,4]
		bms = wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs)
		if type == "DALC":
			plt.plot(bms['values1'])
			plt.plot(bms['values2'])
		else:
			ax = plt.gca()
			ax.set_autoscale_on(False)
			ax.set_aspect('equal')
			plt.plot(bms['values1'],bms['values2'])
		plt.show()
	return
	# 	rhos.append(corr)
	# plt.hist(rhos)
	# plt.show()

# def calcMax():
# 	maxs = []
# 	for i in range(1000):
# 		hurst = 0.2
# 		corr = 0.9
# 		minDiff = 0.2
# 		slopeCoeffs = [2,4]
# 		maxs.append(wfBm(H1= hurst, corr = corr, minDiff = 0.2, slopeCoeffs = slopeCoeffs))
# 		print(i)
# 	return max(maxs)

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