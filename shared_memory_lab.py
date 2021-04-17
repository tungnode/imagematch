from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double

class Point(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]

def modify(n, x, s, A,a_i):
    n.value **= 2
    x.value **= 2
    s.value = s.value.upper()
    # double_array[0] = [1,2,3]
    for a in A:
        a.x **= 2
        a.y **= 2
    A[2] = (1.5,-6.5)   
    a_i[0] = [0.1] 

if __name__ == '__main__':
    lock = Lock()

    n = Value('i', 7)
    x = Value(c_double, 1.0/3.0, lock=False)
    s = Array('c', b'hello world', lock=lock)
    A = Array(Point, [(1.875,-6.25), (-5.75,2.0), (2.375,9.5)], lock=lock)
    a_i = Array(Array('d',1792),10)
    # double_array = Array(List)
    p = Process(target=modify, args=(n, x, s, A,a_i))
    p.start()
    p.join()

    print(n.value)
    print(x.value)
    print(s.value)
    print([(a.x, a.y) for a in A])
    print([(v) for v in a_i[0]])