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