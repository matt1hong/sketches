import random

def encounter(a,b,c):
	p = len(a)
	q = len(b)
	n = len(c)

	solutions = []
	# Possible solutions to the equation k1*p + k2*q = n
	for k1 in range(1,n-q+1):
		if (n - k1*p)%q == 0:
			k2 = (n - k1*p)/q
			if k2 > 0:
				solutions.append((k1,k2))

	for solution in solutions:
		k1 = solution[0]
		k2 = solution[1]
		a1 = a*k1+'2'
		b1 = b*k2+'2'

		memo = [[[0 for z in range(n+1)] for y in range(q*k2+1)] for x in range(p*k1+1)]
		memo[p*k1][q*k2][n] = 1

		for k in range(n-1, -1, -1):
			for i in range(p*k1, -1, -1):
				for j in range(q*k2, -1, -1):
					if a1[i] == b1[j]:
						if a1[i] != c[k]:
							memo[i][j][k] = 0
						else:
							memo[i][j][k] = \
								max(memo[i+1][j][k+1], \
								memo[i][j+1][k+1])
					elif a1[i] == c[k]:
						memo[i][j][k] = memo[i+1][j][k+1]
					elif b1[j] == c[k]:
						memo[i][j][k] = memo[i][j+1][k+1]

		if memo[0][0][0] == 1:
			return True

	return False

def test_encounter():
	for test_sample in range(10):
		# random lengths of a and b
		p = random.randint(1,20)
		q = random.randint(1,20)

		# generate a and b
		a = ''
		b = ''
		for i in range(p):
			a += str(random.randint(0,1))
		for j in range(q):
			b += str(random.randint(0,1))
		print 'a:', a, ' b:', b

		# make c
		c = ''
		a1 = a
		b1 = b
		while a1!='' or b1!='':
			try:
				if random.random()<0.5:
					c += a1[0]
					a1 = a1[1:]
				else:
					c += b1[0]
					b1 = b1[1:]
			except:
				continue

		# test case
		print 'c:', c
		print encounter(a,b,c)

# Output of test_encounter():
# 
# a: 1101  b: 00010011
# c: 110100010011
# True
# 
# a: 001001100101011111  b: 0011010010101
# c: 0000100110010110110001101111011
# True
# 
# a: 10  b: 1001
# c: 101001
# True
# 
# a: 11100011010010  b: 01010001
# c: 0111011000010011010010
# True
# 
# a: 01110101000  b: 1100010010000
# c: 110111001000110000010000
# True
# 
# a: 010000110  b: 110100100
# c: 011000100100111000
# True
# 
# a: 1100111100111011011  b: 1110101010101100
# c: 11001111010111100111011010110101100
# True
# 
# a: 100  b: 110111100011001
# c: 100110111100011001
# True
# 
# a: 1011110000110  b: 11010011
# c: 111001011101001001110
# True
# 
# a: 0010001  b: 0001000011101
# c: 00001010000001011101
# True
