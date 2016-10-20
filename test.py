import _cusgemm
import numpy

a = numpy.random.rand(2,3).astype(numpy.float32)
b = numpy.random.rand(3,4).astype(numpy.float32)

np_res = numpy.dot(a, b)
cu_res =_cusgemm.cusgemm(a, b)

print(np_res)
print(cu_res)
print(numpy.array_equal(np_res, cu_res))
