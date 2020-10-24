import numpy as np

# This algorithm uses strainght line fitting and quadratic curve fitting to make a predictor to predict the number of deaths due to covid 19. The predictor takes in the number of days from 1st May to predict the number of deaths on a particular data.
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
deaths = [73, 71, 83, 72, 195, 126, 89, 103, 95, 128, 97, 87, 122, 134, 100, 103, 120, 157, 134, 140, 132, 148, 137, 147, 154, 146, 170, 194, 175, 265, 193]
m = 22
# Let the hypothese be Y = w0 + w1X where Y is the number of deaths and X is the day.
A = 0
B = 0
C = 0
D = 0
for x in range(0, m):
	A += days[x]
	B += deaths[x]
	C += days[x] * days[x]
	D += deaths[x] * (days[x])

w0 = (B * C - A * D) / (m * C - A * A)
w1 = (A * B - m * D) / (A * A - m * C)

# Calculating error in the last 9 days from the data set.
print("Squared error in linear fitting : ", end = "")
linear_error = 0
for x in range(22, 31):
	linear_error += (w0 + w1 * days[x] - deaths[x]) * (w0 + w1 * days[x] - deaths[x])
print(linear_error)

d1 = -10 #corrosponding to date 20 Apr.
d2 = 41 #corrosponding to date 10 June.

deaths1 = w0 + w1 * d1
deaths2 = w0 + w1 * d2

print("deaths on day 1 : ", end = "")
print(deaths1)
print("deaths on day 2 : ", end = "")
print(deaths2)

print()
days_train = days[:22]
deaths_train = deaths[:22]
col1 = []
col3 = []
for i in days_train:
	col1.append(1)
	col3.append(i * i)
zipconcat = zip(col1, days_train, col3)
MatrixX = list(zipconcat)
Transpose_MatrixX_ = np.transpose(MatrixX)
DotProduct__MatrixX_and_TransposeMatrixX___ = np.dot(Transpose_MatrixX_, MatrixX)
Inverse_DotProduct__MatrixX_and_TransposeMatrixX____ = np.linalg.inv(DotProduct__MatrixX_and_TransposeMatrixX___)
DotProduct_Inverse_DotProduct__MatrixX_and_TransposeMatrixX____and_Transpose_MatrixX__ = np.dot(Inverse_DotProduct__MatrixX_and_TransposeMatrixX____, Transpose_MatrixX_)
MatrixY = []

for i in deaths_train:
	temp = []
	temp.append(i)
	MatrixY.append(temp)

coeff_mat = np.dot(DotProduct_Inverse_DotProduct__MatrixX_and_TransposeMatrixX____and_Transpose_MatrixX__, MatrixY)

w0 = coeff_mat[0][0]
w1 = coeff_mat[1][0]
w2 = coeff_mat[2][0]

# Calculating error in the last 9 days from the data set.
print("Squared error in quadratic fitting : ", end = "")
quadratic_error = 0
for x in range(22, 31):
	quadratic_error += (w0 + w1 * days[x] + w2 * days[x] * days[x] - deaths[x]) * (w0 + w1 * days[x] + w2 * days[x] * days[x] - deaths[x])
print(quadratic_error)

deaths1 = w0 + w1 * d1 + w2 * d1 * d1
deaths2 = w0 + w1 * d2 + w2 * d2 * d2

print("deaths on day 1 : ", end = "")
print(deaths1)
print("deaths on day 2 : ", end = "")
print(deaths2)

print()
if linear_error < quadratic_error:
	print("linear_error is less")
else:
	print("quadratic_error is less")