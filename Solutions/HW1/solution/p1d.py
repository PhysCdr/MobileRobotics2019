import numpy as np

def ortho(m):
	identity_m = np.identity(m.shape[0])
	return np.allclose(np.matmul(m, m.T).flatten(), identity_m.flatten())

def main():
	d = np.array([[2, 2, -1], [2, -1, 2], [-1, 2, 2]])/3

	if ortho(d):
		print('matrix', d, 'is orthogonal')
	else:
		print('matrix', d, 'is not orthogonal')

if __name__ == '__main__':
	main()