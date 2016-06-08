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

return median(bigger[:mid], smaller[-mid:n])