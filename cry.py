def cry(m,t):
	d = len(m)
	n = len(t)

	# Initialize memo table
	memo = [[0 for j in range(n+1)] for i in range(d+1)]

	# Base case
	for book in range(n+1):
		memo[d][book] = sum(t[book:])
	for day in range(d+1):
		memo[day][n] = sum([F**4 for F in m[day:]])

	for i in range(d-1,-1,-1):
		for j in range(n-1,-1,-1):
			tears = []
			for k in range(1,n-j+1):
				Fi = max(m[i] - sum(t[j:j+k]), 0)
				Si = max(sum(t[j:j+k])- m[i], 0)
				tear = Fi**4 + Si
				tears.append(tear + memo[i+1][j+k])
			memo[i][j] = min(tears)
	
	return memo[0][0]
