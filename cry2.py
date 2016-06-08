# def cry(m,t):
# 	d = len(m)
# 	m = m+[0]

# 	# Initialize memo table
# 	memo = [[0 for j in range(len(t)+1)] for i in range(d+1)]

# 	# Base case
# 	for day in range(d+1):
# 		memo[day][len(t)] = (sum(m[day:]))**4

# 	for book in range(len(t)+1):
# 		memo[d][book] = sum(t[book:])

# 	print memo

# 	T = [0] * d

# 	for i in range(d-2,-1,-1):
# 		T = 0
# 		for j in range(len(t)-1,-1,-1):
# 			value1 = memo[i][j+1] + \
# 				max((m[i] - T[i] + t[j])**4, T[i] + t[j] - m[i])
# 			value2 = memo[i+1][j+1] + \
# 				max((m[i+1] - T[i+1] + t[j])**4, T[i+1] + t[j] - m[i+1])

# 			if value1 < value2:
# 				T[i] += t[j]
# 				memo[i][j] = value1
# 			else:
# 				T[i+1] += t[j]
# 				memo[i][j] = value2
# 			print T

# 	return memo