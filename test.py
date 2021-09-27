import numpy as np
from numpy.lib.npyio import load
from scipy import sparse
import math
from pprint import pprint
'''
a = np.array([[0,0,0,0,0,0,1,0]])

b = np.array([[0,0,0,0,0,0,0,1]])
d = np.array([[1],[2]])


ac = sparse.coo_matrix(a)
bc = sparse.coo_matrix(b)
mat = ac.dot(bc.transpose())
print(mat.toarray())


a_csr = sparse.csr_matrix(a)


cc = sparse.vstack((ac,bc))
#print(cc.toarray())
'''
'''
a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.int16)
a = sparse.coo_matrix(a)
print(a)
sparse.save_npz('my.npz',a)
loaded = sparse.load_npz('my.npz')
print('\n{}'.format(loaded))
'''
a = np.zeros([8,1])
b = np.ones([8,1])
print(np.append(a,b,axis=1))
