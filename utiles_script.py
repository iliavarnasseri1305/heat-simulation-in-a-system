import math
import random
import matplotlib.pyplot as plt 

class AlgebraicFunc:
	def __init__(self, function_array):
		"""
		self.array = [(c1, p1), (c2, p2), (c3, p3), ..., (cn, pn)]
		"""
		self.array = function_array[:]

	def __call__(self, x):
		s = 0
		for c, p in self.array:
			s += c * x ** p

		return s 

	def __add__(self, other):
		new = {}
		for c, p in self.array + other.array:
			if p in new.keys() :
				new[p] += c

			else:
				new.update({p : c})

		return AlgebraicFunc(new.items())

	def __sub__(self, other):
		return self + (-1) * other

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return AlgebraicFunc([other * i for i in other.array])

		elif isinstance(other, AlgebraicFunc):
			new = []
			for c1, p1 in self.array:
				for c2, p2 in other.array:
					f = AlgebraicFunc([c1 * c2, p1 + p2])
					new.append(f)

			s = AlgebraicFunc([0, 0])
			for i in new:
				s += i  

			return s 

	def derivative(self):
		new = [(c * p, p - 1) for c, p in self.array]
		return AlgebraicFunc(new[:])

	def antiderivative(self):
		new = [(c / (p + 1), p + 1) for c, p in self.array]
		return AlgebraicFunc(new[:])

	def higherOrderDerivative(self, n):
		func = self
		for i in range(n):
			func = func.derivative()

		return func 

	def higherOrderAntiderivative(self, n):
		func = self
		for i in range(n):
			func = func.antiderivative()

		return func

	def graph(self, interval, step=.01):
		x_array, y_array = [], []
		i = interval[0]
		while i <= interval[1]:
			x_array.append(i)
			y_array.append(self(i))
			i += step

		return x_array, y_array


class Function:
	def __init__(self, func):
		self.func = func 

	def __add__(self, other):
		return Function(lambda x : self.func(x) + other.func(x))

	def __sub__(self, other):
		return Function(lambda x : self.func(x) - other.func(x))

	def __mul__(self, other):
		return Function(lambda x : self.func(x) * other.func(x))

	def __truediv__(self, other):
		return Function(lambda x : self.func(x) / other.func(x))

	def __call__(self, x):
		return self.func(x)

	def graph(self, interval, step=.01):
		x_array, y_array = [], []
		i = interval[0]
		while i <= interval[1]:
			x_array.append(i)
			y_array.append(self(i))
			i += step
			
		return x_array, y_array

class Matrix:
	def __init__(self, array):
		self.array = array[:]
		self.dim = [len(self.array), len(self.array[0])]

	def __mul__(self, other):
		new_array = [[0] * self.dim[0]] * other.dim[1]
		for i in range(len(self.array)):
 
	# iterating by column by B
			for j in range(len(other.array[0])):
 
		# iterating by rows of B
				for k in range(len(other.array)):
					new_array[i][j] += A[i][k] * B[k][j]
		
		return Matrix(new_array)

class Vector:
	def __init__(self, arr):
		self.arr = arr 
		self.dim = len(self.arr)

	def __add__(self, other):
		if self.dim != other.dim:
			raise Exception(ValueError, "the two vectors should be of the same dimention")

		elif isinstance(other, Vector):
			new_array = [0] * self.dim
			for i in range(len(new_array)):
				new_array[i] += (self.arr[i] + other.arr[i])

			return Vector(new_array)

		elif isinstance(other, (int, float)):
			new_array = [other * i for i in self.array]
			return Vector(new_array)

	def __mul__(self, other):
		return sum([self.arr[i] * other.arr[i] for i in range(self.dim)])

	def __sub__(self, other):
		return self + (-1) * other

	def cross(self, other):
		if self.dim != 3 or self.dim != other.dim:
			raise Exception(ValueError, "the two vectors should be of the same dimention = 3")

		else:
			return Vector([self.arr[2]*other.arr[3] - self.arr[3]*other.arr[2],
			 				self.arr[3]*other.arr[1] - self.arr[1]*other.arr[3],
			  				self.arr[1]*other.arr[2] - self.arr[2]*other.arr[1]
			  				])


def derivative(func):
	dx = .01
	return lambda p : (func(p + dx) - func(p - dx)) / (2 * dx)

def integral(func, dx=.001):

	def der(a, b):
		s = 0 
		i = a
		while i <= b:
			s += func(i) * dx
			i += dx
		return s

	return der  

def higher_order_derivative(func, n):
	if n == 0 :
		return func 

	elif n > 0:
		return derivative(higherOrderDerivative(func, n - 1))

def taylorExpansion(function, point, n):
	
	def expression(x):
		s = 0
		for i in range(n):
			s += (((x - point) ** i) / (math.factorial(i))) * higher_order_derivative(function, i)(x)

		return s 

	return expression

def graph(function, interval, step=.01, autoscale=False):
	if isinstance(function, (Function, AlgebraicFunc)):
		gr_data = function.graph(interval[:])
		xlim_0, xlim_1 = min(gr_data[0]), max(gr_data[0])
		ylim_0, ylim_1 = min(gr_data[1]), max(gr_data[1])
		if not autoscale:
			plt.xlim(xlim_0, xlim_1)
			plt.ylim(ylim_0, ylim_1)
			plt.autoscale(False)


	else:
		def gr_data_calculate(function, interval, step=.01):
			x_array, y_array = [], []
			i = interval[0]
			while i <= interval[1]:
				x_array.append(i)
				y_array.append(function(i))
				i += step
				
			return x_array, y_array

		gr_data = gr_data_calculate(function, interval[:])
		xlim_0, xlim_1 = min(gr_data[0]), max(gr_data[0])
		ylim_0, ylim_1 = min(gr_data[1]), max(gr_data[1])
		if not autoscale:
			plt.xlim(xlim_0, xlim_1)
			plt.ylim(ylim_0, ylim_1)
			plt.autoscale(False)

		return plt.plot(gr_data[0], gr_data[1])

def newtons_method(function, domain, req):
	x = (domain[1] + domain[0]) / 2
	for i in range(req):
		if derivative(function)(x) == 0:
			x = (domain[0] + x) / 2
			continue; 

		x -= function(x) / derivative(function)(x)

	return x 

def mean(data):
	return sum(data) / len(data)

def variance(data):
	mean_flt = mean(data)
	return sum([(i ** 2) / len(data) for i in data]) - mean_flt ** 2

def devieation_from_standard(data):
	return (variance(data)) ** .5

def split_interval(interval, section):
	step = (interval[1] - interval[0]) / section
	array = []
	i = interval[0]
	while i <= interval[-1]:
		array.append(i)
		i += step

	return array[:]

def continuous_mean(function, interval):
	return (1 / (interval[1] - interval[0])) * integral(function)(interval[0], interval[1])

def continuous_variance(function, interval, dx=.001):
	return abs(continuous_mean((lambda x : function(x)**2), interval) - (continuous_mean(function, interval))**2)