import numpy as np
wines = np.genfromtxt("Housing Price data set.csv", delimiter=",", skip_header=1)

Price=[]
NoOfBathrooms=[]
NoOfBedrooms=[]
FloorArea=[]

for i in range(len(wines)):
	Price.append(wines[i][1])
for i in range(len(wines)):
	FloorArea.append(wines[i][2])
for i in range(len(wines)):
	NoOfBedrooms.append(wines[i][3])
for i in range(len(wines)):
	NoOfBathrooms.append(wines[i][4])

FloorAreaTrain = FloorArea[:382]
NoOfBathroomsTrain = NoOfBathrooms[:382]
NoOfBedroomsTrain = NoOfBedrooms[:382]
PriceTrain = Price[:382]

Col1 = []

for i in range(len(FloorAreaTrain)):
	Col1.append(1)

zipconcat = zip(Col1, FloorAreaTrain, NoOfBedroomsTrain, NoOfBathroomsTrain)
MatrixX = list(zipconcat)
Transpose_MatrixX_ = np.transpose(MatrixX)
DotProduct__MatrixX_and_TransposeMatrixX___ = np.dot(Transpose_MatrixX_, MatrixX)
Inverse_DotProduct__MatrixX_and_TransposeMatrixX____ = np.linalg.inv(DotProduct__MatrixX_and_TransposeMatrixX___)
DotProduct_Inverse_DotProduct__MatrixX_and_TransposeMatrixX____and_Transpose_MatrixX__ = np.dot(Inverse_DotProduct__MatrixX_and_TransposeMatrixX____, Transpose_MatrixX_)
MatrixY = []

for i in PriceTrain:
	temp = []
	temp.append(i)
	MatrixY.append(temp)

coeff_mat = np.dot(DotProduct_Inverse_DotProduct__MatrixX_and_TransposeMatrixX____and_Transpose_MatrixX__, MatrixY)

# Now finding error without normalisation.
Y_pred = []
for i in range(383, 546):
  Y_pred.append(int(coeff_mat[0][0] + coeff_mat[1][0] * FloorArea[i] + coeff_mat[2][0] * NoOfBedrooms[i] + coeff_mat[3][0] * NoOfBathrooms[i]))
error =0
for i in range(163):
    error = error + abs((Y_pred[i] - Price[i + 382]) / Price[i + 382])
error = error / 163
print(error * 100)

# Now finding error with normalisation
constant = np.identity(4, dtype=float)
MatrixTemp = np.dot(Transpose_MatrixX_, MatrixY)
constant[0][0] = 0
for i in range(30):
	Bar_DotProduct__MatrixX_and_TransposeMatrixX____ = np.add(DotProduct__MatrixX_and_TransposeMatrixX___, constant * i)
	Inverse_Bar_DotProduct__MatrixX_and_TransposeMatrixX_____ = np.linalg.inv(Bar_DotProduct__MatrixX_and_TransposeMatrixX____)
	coeff_mat = np.dot(Inverse_Bar_DotProduct__MatrixX_and_TransposeMatrixX_____, MatrixTemp)
	Y_pred = []
	for i in range(383, 546):
	  Y_pred.append(int(coeff_mat[0][0] + coeff_mat[1][0] * FloorArea[i] + coeff_mat[2][0] * NoOfBedrooms[i] + coeff_mat[3][0] * NoOfBathrooms[i]))
	errort = 0
	for i in range(163):
	    errort = errort + abs((Y_pred[i] - Price[i + 382]) / Price[i + 382])
	errort = error / 163
	error = min(error, errort)
print(error)