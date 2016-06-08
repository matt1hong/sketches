def median(A,B):
	n = len(A)

	if n == 1:
		if A[0] < B[0]:
			return A[0]
		else:
			return B[0]

	mid = (n - 1) / 2

	if A[mid] > B[mid]:
		bigger = A
		smaller = B
	else:
		bigger = B
		smaller = A

	if n == 2:
		return median([bigger[0]], [smaller[1]])

	return median(bigger[:mid], smaller[-mid:n])

def parse(p):
	if len(p) == 1:
		# print p[0]
		return p[0]
	for i in range(1,len(p)):
		print (parse(p[0:i]), parse(p[i:len(p)]))