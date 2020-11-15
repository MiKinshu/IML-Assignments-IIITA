import numpy as np
import random
import math
import matplotlib.pyplot as plt

rows, cols = 10, 10
alpha = 0.1
class neuron:
	def __init__(self, x, y, weight_x, weight_y):
		self.x = x
		self.y = y
		self.weight_x = weight_x
		self.weight_y = weight_y

	def update(self,x,y):
		self.weight_x = self.weight_x + alpha*(x - self.weight_x)
		self.weight_y = self.weight_y + alpha*(y - self.weight_y)

	def distance(self,x,y):
		return math.sqrt((x-self.x)**2+(y-self.y)**2)

	def print(self):
		print("(" + str(self.x) + "," + str(self.y) + ")=(" + str(self.weight_x)+","+str(self.weight_y)+")", end="")


def initialize_Kohonen_network():
	Kohonen_network = [[neuron((i / 10) * 2 - 1, (j / 10) * 2 - 1, random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for i in range(cols)] for
					   j in range(rows)]
	return Kohonen_network

def update(Kohonen_network,x,y):
	min_x,min_y = 0,0
	min_ = 100.0
	for i_x in range(0, cols):
		for i_y in range(0, rows):
			if min_ > Kohonen_network[i_x][i_y].distance(x,y):
				min_ = Kohonen_network[i_x][i_y].distance(x,y)
				min_x, min_y = i_x, i_y
	Kohonen_network[min_x][min_y].update(x,y)

def train(Kohonen_network):
	for i in range(0,1500):
		update(Kohonen_network,random.uniform(-1,1),random.uniform(-1,1))
	return  Kohonen_network

def test(Kohonen_network,x,y):
	min_x,min_y = 0,0
	min_ = 100.0
	for i_x in range(0, cols):
		for i_y in range(0, rows):
			if min_ > Kohonen_network[i_x][i_y].distance(x,y):
				min_ = Kohonen_network[i_x][i_y].distance(x,y)
				min_x, min_y = i_x, i_y
	Kohonen_network[min_x][min_y].print()

def plot(Kohonen_network):
	x = []
	y = []
	for x_i in range(0, cols):
		for y_i in range(0, rows):
			x.append(Kohonen_network[x_i][y_i].weight_x)
			y.append(Kohonen_network[x_i][y_i].weight_y)
	plt.plot(x,y)
	plt.show()

def print_Kohonen_network(Kohonen_network):
	print("##################################################################")
	for x in range(0, cols):
		for y in range(0, rows):
			Kohonen_network[x][y].print()
			print(" ", end="")
		print()
	print("##################################################################")

Kohonen_network = initialize_Kohonen_network()
Kohonen_network = train(Kohonen_network)
test(Kohonen_network,-0.8,0.8)
# plot(Kohonen_network)