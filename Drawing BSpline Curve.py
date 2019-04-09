import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
global root
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
def insert_point():
    var = t1.get("1.0", 'end-1c')
    var = var.replace('The points:', '')
    var = var.replace(',', ' ')
    var = var.replace('(', '')
    var = var.replace(')', '')
    L = var.split()
    X = list(map(float, L[0:-1:2]))
    Y = list(map(float, L[1:len(L):2]))
    Main(X=X, Y=Y)

def read_file():
    var = t.get("1.0", 'end-1c')
    INPUT_PATH = var.replace('The input path:', '')
    # input data file
    D_input = np.loadtxt(INPUT_PATH)
    print(D_input)  # show the D input points
    string = ''
    for i in range(len(D_input)):
        string = str(string) + str(tuple(D_input[i])) + ','
    t1.insert("2.0", string)

epsilon = np.finfo(np.float32).eps
def weight_n(u, i, j, knotvector):
    if j == 0:
        if knotvector[i] <= u < knotvector[i+1] and knotvector[i] < knotvector[i+1]:
            return 1
        else:
            return 0
    if knotvector[i+j] - knotvector[i] == 0 and u - knotvector[i] == 0:
        weight1 = 0
    else:
        weight1 = (u - knotvector[i]) / (knotvector[i+j] - knotvector[i] + epsilon)
    if knotvector[i+j+1] - knotvector[i+1] == 0 and knotvector[i+j+1] - u == 0:
        weight2 = 0
    else:
        weight2 = (knotvector[i+j+1] - u) / (knotvector[i+j+1] - knotvector[i+1] + epsilon)
    return weight1*weight_n(u=u, i=i, j=j-1, knotvector=knotvector) + weight2 * weight_n(u=u, i=i+1, j=j-1, knotvector=knotvector)

def lengthBetweenPoints(point1, point2):
    distance = np.square(point1[:, 0] - point2[:, 0]) + np.square(point1[:, 1] - point2[:, 1])
    return np.sqrt(distance)

def ChordLengthParameterization(points):
    point1 = np.array(points[:-1:])
    point2 = np.array(points[1::])
    dis = lengthBetweenPoints(point1=point1, point2=point2)
    dis = np.absolute(dis)
    denominator = np.sum(a=dis, axis=-1)
    numerator = np.cumsum(a=dis, axis=-1, dtype=np.float)
    weights = numerator / denominator
    weights = list(weights)
    weights.append(0)
    weights = sorted(weights)
    weights = np.round(a=weights, decimals=12)
    return weights

def u1func(u, i, knot_vector):
    u1 = (u - knot_vector[i]) ** 3 / float(
        (knot_vector[i + 1] - knot_vector[i]) * (knot_vector[i + 2] - knot_vector[i]) * (
                    knot_vector[i + 3] - knot_vector[i]) + epsilon)
    return u1

def u2func(u, i, knot_vector):
    tmp1 = (knot_vector[i + 2] - u) * (u - knot_vector[i]) ** 2 / float(
        (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i]) * (
                knot_vector[i + 2] - knot_vector[i]) + epsilon)
    tmp2 = (knot_vector[i + 3] - u) * (u - knot_vector[i]) * (u - knot_vector[i + 1]) / float(
        (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                knot_vector[i + 3] - knot_vector[i]) + epsilon)
    tmp3 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) ** 2 / float(
        (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
    u2 = tmp1 + tmp2 + tmp3
    return u2

def u3func(u, i, knot_vector):
    tmp1 = (u - knot_vector[i]) * (knot_vector[i + 3] - u) ** 2 / float(
        (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                knot_vector[i + 3] - knot_vector[i]) + epsilon)
    tmp2 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) * (knot_vector[i + 3] - u) / float(
        (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
    tmp3 = (u - knot_vector[i + 2]) * (knot_vector[i + 4] - u) ** 2 / float(
        (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 2]) * (
                knot_vector[i + 4] - knot_vector[i + 1]) + epsilon)
    u3 = tmp1 + tmp2 + tmp3
    return u3

def u4func(u, i, knot_vector):
    u4 = (knot_vector[i + 4] - u) ** 3 / float(
        (knot_vector[i + 4] - knot_vector[i + 3]) * (knot_vector[i + 4] - knot_vector[i + 2]) * (
                    knot_vector[i + 4] - knot_vector[i + 1]) + epsilon)
    return u4

def Ni3(u, i, knot_vector):
    global epsilon
    u1 = 0
    u2 = 0
    u3 = 0
    u4 = 0
    if knot_vector[i] <= u <= knot_vector[i+1] and knot_vector[i] < knot_vector[i+1]:
        u1 = (u-knot_vector[i])**3 / float((knot_vector[i+1]-knot_vector[i])*(knot_vector[i+2]-knot_vector[i])*(knot_vector[i+3]-knot_vector[i]) + epsilon)
    if knot_vector[i+1] <= u <= knot_vector[i+2] and knot_vector[i+1] < knot_vector[i+2]:
        tmp1 = (knot_vector[i + 2] - u) * (u - knot_vector[i]) ** 2 / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i]) * (
                        knot_vector[i + 2] - knot_vector[i]) + epsilon)
        tmp2 = (knot_vector[i + 3] - u) * (u - knot_vector[i]) * (u - knot_vector[i + 1]) / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i]) + epsilon)
        tmp3 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) ** 2 / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
        u2 = tmp1 + tmp2 + tmp3
    if knot_vector[i+2] <= u <= knot_vector[i+3] and knot_vector[i+2] < knot_vector[i+3]:
        tmp1 = (u - knot_vector[i]) * (knot_vector[i + 3] - u)**2 / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i]) + epsilon)
        tmp2 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) * (knot_vector[i + 3] - u) / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
        tmp3 = (u - knot_vector[i + 2]) * (knot_vector[i + 4] - u)**2 / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 2]) * (
                        knot_vector[i + 4] - knot_vector[i + 1]) + epsilon)
        u3 = tmp1 + tmp2 + tmp3
    if knot_vector[i+3] <= u <= knot_vector[i+4] and knot_vector[i+3] < knot_vector[i+4]:
        u4 = (knot_vector[i+4] - u)**3 / float((knot_vector[i+4] - knot_vector[i+3])*(knot_vector[i+4] - knot_vector[i+2])*(knot_vector[i+4]-knot_vector[i+1]) + epsilon)
    num = np.count_nonzero(a=[u1, u2, u3, u4])
    if u1 + u2 + u3 + u4 == 0:
        return 0
    return (u1 + u2 + u3 + u4) / num


def fprime(u, i, knotvector, u0):
    u1derative = 0
    u2derative = 0
    u3derative = 0
    u4derative = 0
    if knotvector[i] <= u0 <= knotvector[i+1] and knotvector[i] < knotvector[i+1]:
        u1_exp = sym.diff(u1func(u=u, i=i, knot_vector=knotvector), u, 2)
        print('the u1_exp:', u1_exp)
        u1derative = u1_exp.evalf(subs={u: u0})
    if knotvector[i+1] <= u0 <= knotvector[i+2] and knotvector[i+1] < knotvector[i+2]:
        u2_exp = sym.diff(u2func(u=u, i=i, knot_vector=knotvector), u, 2)
        print('the u2_exp:', u2_exp)
        u2derative = u2_exp.evalf(subs={u: u0})
    if knotvector[i + 2] <= u0 <= knotvector[i + 3] and knotvector[i+2] < knotvector[i+3]:
        u3_exp = sym.diff(u3func(u=u, i=i, knot_vector=knotvector), u, 2)
        print('the u3_exp:', u3_exp)
        u3derative = u3_exp.evalf(subs={u: u0})
    if knotvector[i + 3] <= u0 <= knotvector[i + 4] and knotvector[i+3] < knotvector[i+4]:
        u4_exp = sym.diff(u4func(u=u, i=i, knot_vector=knotvector), u, 2)
        print('the u4_exp:', u4_exp)
        u4derative = u4_exp.evalf(subs={u: u0})
    num = np.count_nonzero(a=[u1derative, u2derative, u3derative, u4derative])
    return (u1derative + u2derative + u3derative + u4derative) / float(num)

def BSpline(P, U, knot_vector):
    points = []
    for u in U:
        tmp = [Ni3(u=u, i=i, knot_vector=knot_vector) for i in range(len(P))]
        tmp = np.reshape(a=tmp, newshape=(-1, 1))
        tmp = np.sum(a=tmp * P, axis=0)
        points.append(tmp)
    return points



def Main(X, Y):

    ###########################################
    number = len(X)
    N = number - 1
    points = list(zip(X, Y))
    print(points)
    ###########################################

    ###########################################
    #generate knot vector in useful range
    # k to n+1
    # k is degree of the curve, n is number of control points
    k = 3
    # uniform parameterization
    knot_vector1 = np.arange(0, 1+1/N, 1/N, np.float)
    knot_vector1 = np.round(a=knot_vector1, decimals=12)
    # Chord length parameterization
    knot_vector2 = ChordLengthParameterization(points=points)
    n = k - 1 + N
    m = n + k + 1
    # recovering the knot vectors
    a1 = np.repeat(a=knot_vector1[0], repeats=k)
    a2 = np.repeat(a=knot_vector1[-1], repeats=k)
    knot_vector1 = np.concatenate([a1, knot_vector1, a2], axis=-1)
    knot_vector2 = np.concatenate([a1, knot_vector2, a2], axis=-1)
    knot_vector2[0] = knot_vector2[0] - 3 * epsilon
    knot_vector2[1] = knot_vector2[1] - 2 * epsilon
    knot_vector2[2] = knot_vector2[2] - 1 * epsilon
    knot_vector2[-1] = knot_vector2[-1] + 3 * epsilon
    knot_vector2[-2] = knot_vector2[-2] + 2 * epsilon
    knot_vector2[-3] = knot_vector2[-3] + 1 * epsilon
    matrix = np.zeros(shape=(N + 1 + 2, N+2+1))
    matrix1 = np.zeros(shape=(N + 1 + 2, N + 2 + 1))
    for i in range(N+1):
        weight_k = weight_n(u=knot_vector1[i+k], i=i, j=3, knotvector=knot_vector2)
        u1 = Ni3(u=knot_vector1[i+k], i=i, knot_vector=knot_vector1)
        weight_k1 = weight_n(u=knot_vector1[i+k], i=i + 1, j=3, knotvector=knot_vector2)
        u2 = Ni3(u=knot_vector1[i+k], i=i + 1, knot_vector=knot_vector1)
        weight_k2 = weight_n(u=knot_vector1[i+k], i=i + 2, j=3, knotvector=knot_vector2)
        u3 = Ni3(u=knot_vector1[i+k], i=i + 2, knot_vector=knot_vector1)
        index = np.arange(i, i + k, 1, np.int)
        matrix[i+1, index] = [weight_k, weight_k1, weight_k2]
        matrix1[i+1, index] = [u1, u2, u3]

    # computing the gradient
    # h is precise interval
    h = 0.000001
    u0 = knot_vector1[0+k]
    u = sym.symbols('u')
    t1 = fprime(u=u, i=0, knotvector=knot_vector1, u0=u0)
    t2 = fprime(u=u, i=1, knotvector=knot_vector1, u0=u0)
    t3 = fprime(u=u, i=2, knotvector=knot_vector1, u0=u0)

    matrix1[0, 0] = t1
    matrix1[0, 1] = t2
    matrix1[0, 2] = t3
    u1 = knot_vector1[N+k]

    t11 = fprime(u=u, i=N, knotvector=knot_vector1, u0=u1)
    t12 = fprime(u=u, i=N+1, knotvector=knot_vector1, u0=u1)
    t13 = fprime(u=u, i=N+2, knotvector=knot_vector1, u0=u1)
    matrix1[-1, -3] = t11
    matrix1[-1, -2] = t12
    matrix1[-1, -1] = t13
    matrix1 = np.round(a=matrix1, decimals=3)
    D = np.array(points)
    D = np.concatenate([[(0, 0)], D, [(0, 0)]], axis=0)
    P1 = np.linalg.solve(a=matrix1+epsilon, b=D)
    P1 = np.round(a=P1, decimals=8)
    xx = np.linspace(0, 1, 50)
    points = BSpline(P=P1, U=xx, knot_vector=knot_vector1)
    points = np.array(points)

    ############################################################################
    # drawing
    ##################################################################
    # output file
    outputfile(OUTPUT_PATH='Output.txt', P=P1, knot_vector=knot_vector1, degree=3)
    ##################################################################
    f = Figure(figsize=(5, 5), dpi=100)
    a = f.add_subplot(111)
    a.plot(points[:, 0], points[:, 1], 'b-', lw=4, alpha=0.7, label='BSpline')

    canvas = FigureCanvasTkAgg(f, root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    ###########################################


def outputfile(OUTPUT_PATH, P, knot_vector, degree=3):
    # -------#
    ## G. Output the file
    f = open(OUTPUT_PATH, 'w')

    # output the title
    f.write('cubic.txt ---')
    f.write('\n\n')

    # output the degree
    f.write(str(degree))
    f.write('\n')

    # output the number of control points
    f.write(str(len(P)))
    f.write('\n\n')

    # output the knot vector
    for i in range(len(knot_vector)):
        f.write(str(knot_vector[i]))
        f.write('\t')
    f.write('\n\n')

    # output the control points
    for i in range(len(P)):
        f.write(str(P[i][0]))
        f.write('\t')
        f.write(str(P[i][1]))
        f.write('\n')
    f.close()


if __name__ == "__main__":
    global root
    root = tk.Tk()
    root.title('Bspline Curve')
    # the window size
    root.geometry('200x200')
    b1 = tk.Button(root, text="draw curve", width=15, height=2, command=insert_point)
    b1.pack()
    b2 = tk.Button(root, text="read file", width=15, height=2, command=read_file)
    b2.pack()
    t = tk.Text(root, height=2)
    t.insert("1.0", 'The input path:point.txt')
    t.pack()
    t1 = tk.Text(root, height=2)
    t1.insert("1.0", 'The points:')
    t1.pack()
    root.mainloop()  # 进入消息循环