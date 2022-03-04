import matplotlib.pyplot as plt
import utiles_script as ut 
import random  
import math

#### ALL UNITS ARE IN SI ####
#### CONSTANTS ####
NA = 6.02214076 * 10**23

### OBJECTS ###
class Particle:
	def __init__(self, function, mass, domain_of_ocillation, c):
		self.function = function
		self.mass = mass
		self.domain_of_ocillation = domain_of_ocillation
		self.c = c

	def position(self, t):
		return self.function(t)

	def velocity(self, t):
		return ut.derivative(self.function)(t)

	def acceleration(self, t):
		return ut.higher_order_derivative(self.function, 2)(t)

### FUNCTIONS ###

#### it creates water molecules
def create_random_sample(number_of_particles):
	array = []
	for i in range(int(number_of_particles)):
		initial_position = random.randint(0, 10)
		domain_of_ocillation = random.random()
		T = random.randint(1, 10);
		omega = 2 * math.pi / T
		function = ut.Function(lambda x : initial_position + domain_of_ocillation * math.cos(omega * x)) 
		particle = Particle(function, 2.988 * 10**(-26), domain_of_ocillation, 4200)
		array.append(particle)

	return array

def compute_mean_var(array_of_particles):
	variance_array = []
	for p in array_of_particles:
		variance_array.append(ut.continuous_variance(p.function, [-p.domain_of_ocillation, p.domain_of_ocillation], dx=.01))

	return sum(variance_array) / len(variance_array)

### assuming the c component for all particles is equal
def compute_temp(array_of_particles, t=1):
	kinetic_energy_array = []
	for p in array_of_particles:
		kinetic_energy_array.append(.5 * p.mass * (p.velocity(t))**2)

	total_energy = sum(kinetic_energy_array)
	total_mass = sum([p.mass for p in array_of_particles])
	c = array_of_particles[0].c

	return total_energy / (total_mass * c)

### running the simulation ###

array_of_array_of_particles = [create_random_sample(500) for i in range(0, 1000)] # 1000 water particles
print("particle array created successfully.")
temp_array = [compute_temp(arr) for arr in array_of_array_of_particles]
print("temperature array created successfully.")
var_array = [compute_mean_var(arr) for arr in array_of_array_of_particles]
print("variance array created successfully.")
plt.scatter(temp_array, var_array, s=5)
print("plot created successfully.")
plt.show()
