import networkx as nx
import numpy as np
import pandas as pd


def parse_data():
	names = [name.strip() for name in open('data/names.txt')]

	data = np.array([
		[int(v) for v in line.split()]
		for line in open('data/chowliu-input.txt')
	])

	return names, data

def mutual_information(x, y):
	(size,) = x.shape
	mi = 0
	for x_i in (0, 1):
		for y_i in (0, 1):
			p_x_y = np.sum((x == x_i) & (y == y_i))
			if p_x_y > 0:
				p_x = np.sum(x == x_i) / size
				p_y = np.sum(y == y_i) / size
				mi += p_x_y * np.log(p_x_y) - np.log(p_x) - np.log(p_y)
	return mi


def calculate_adjacency_matrix(data):
	pairwise_mi = np.empty((data.shape[1], data.shape[1]))
	for i in range(data.shape[1]):
		for j in range(i, data.shape[1]):
			pairwise_mi[i, j] = mutual_information(data[:, i], data[:, j])
	return pairwise_mi


_, data = parse_data()
adj = calculate_adjacency_matrix(data)
G = nx.from_numpy_array(adj)
T = nx.maximum_spanning_tree(G)

print("MARKOV")
print(T.number_of_nodes())
print(' '.join('2' for _ in range(T.number_of_nodes())))
print(T.number_of_edges() + T.number_of_nodes())
# Factor specifications
for i in range(T.number_of_nodes()):
	print("1 {}".format(i))
for (i, j) in T.edges:
	print("2 {} {}".format(i, j))
# Factor tables
prob_x_equals_1 = np.sum(data, axis=0) / data.shape[0]
for i, p in enumerate(prob_x_equals_1):
	print(i)
	print(" {0:.3f} {0:.3f}".format(1 - p, p))

for (i, j) in T.edges:
	x, y = data[i], data[j]
	size = x.shape[0]
	print(4)
	for x_i in (0, 1):
		row = []
		for y_i in (0, 1):
			p_x_y = np.sum((x == x_i) & (y == y_i)) / size
			if p_x_y > 0:
				p_x = np.sum(x == x_i) / size
				p_y = np.sum(y == y_i) / size
				row.append(p_x_y / (p_x * p_y))
			else:
				row.append(0.0)
		print(''.join(" {0:.3f}".format(v) for v in row))


