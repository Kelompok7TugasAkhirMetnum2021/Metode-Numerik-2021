#!/usr/bin/env python
# coding: utf-8

# In[2]:


##### INPUT #####
##### MODUL 2 #####
##### Metode Setengah Interval #####
def SetengahInterval (X1, X2, a, b, c, d):
    X1 = X1
    X2 = X2
    a = a
    b = b
    c = c
    d = d
    error = 1
    iterasi = 0
    while(error > 0.0001):
        iterasi +=1
        FXi = (float(a*((X1)**3)))+(float(b*((X1)**2)))+(c*X1)+d
        FXii = (float(a*((X2)**3)))+(float(b*((X2)**2)))+(c*X2)+d
        Xt = (X1+X2)/2
        FXt = (float(a*((Xt)**3)))+(float(b*((Xt)**2)))+(c*Xt)+d
        if FXi*FXt > 0:
            X1 = Xt
        elif FXi*FXt < 0:
            X2 = Xt
        else:
            print("Akar Penyelesaian: ", Xt)
        if FXt < 0:
            error = FXt*(-1)
        else:
            error = FXt
        if iterasi > 100:
            print("Angka Tak Hingga")
            break
        print(iterasi, "|", FXi, "|", FXii, "|", Xt, "|", FXt, "|", error)
    print("Jumlah Iterasi = ", iterasi)
    print("Akar Persamaan = ", Xt)
    print("Toleransi Error = ", error)

##### Metode Interpolasi Linear #####
def InterpolasiLinear (X1, a, b, c, d):
    X1 = X1
    X2 = X1 + 1
    a = a
    b = b
    c = c
    d = d
    error = 1
    iterasi = 0
    while(error > 0.0001):
        iterasi +=1
        FX1 = (float(a*((X1)**3)))+(float(b*((X1)**2)))+(c*X1)+d
        FX2 = (float(a*((X2)**3)))+(float(b*((X2)**2)))+(c*X2)+d
        Xt = X2-((FX2/(FX2-FX1)))*(X2-X1)
        FXt = (float(a*((Xt)**3)))+(float(b*((Xt)**2)))+(c*Xt)+d
        if FXt*FX1 > 0:
            X2 = Xt
            FX2 = FXt
        else:
            X1 = Xt
            FX1 = FXt
        if FXt < 0:
            error = FXt*(-1)
        else:
            error = FXt
        if iterasi > 500:
            print("Angka Tak Hingga")
            break
        print(iterasi, "|", FX1, "|", FX2, "|", Xt, "|", FXt, "|", error)
    print("Jumlah Iterasi = ", iterasi)
    print("Akar Persamaan = ", Xt)
    print("Toleransi Error = ", error)

##### Metode Newton-Rhapson #####
def NewtonRhapson (X1, a, b, c, d):
    X1 = X1
    a = a
    b = b
    c = c
    d = d
    iterasi = 0
    akar = 1
    while (akar > 0.0001):
        iterasi += 1
        Fxn = (float(a*((X1)**3)))+(float(b*((X1)**2)))+(c*X1)+d
        Fxxn = (float((a*3)*X1)**2)+(float((b*2)*X1))+(c)
        xnp1 = X1-(Fxn/Fxxn)
        fxnp1 = (a*(xnp1**3))+(b*(xnp1**2))-(c*xnp1)+d
        Ea = ((xnp1-X1)/xnp1)*100
        if Ea < 0.0001:
            X1 = xnp1
            akar = Ea*(-1)
        else:
            akar = xnp1
            print("Nilai akar adalah: ", akar)
            print("Nilai error adalah: ", Ea)
        if iterasi > 100:
            break   
        print(iterasi, "|", X1, "|", xnp1, "|", akar, "|", Ea)
    print("Jumlah Iterasi = ", iterasi)
    print("Akar Persamaan = ", xnp1)
    print("Toleransi Error = ", akar)

##### Metode Secant #####
def Secant (X1, a, b, c, d):
    X1 = X1
    X2 = X1 - 1
    a = a
    b = b
    c = c
    d = d
    error = 1
    iterasi = 0
    while(error > 0.0001):
        iterasi +=1
        FX1 = (float(a*((X1)**3)))+(float(b*((X1)**2)))+(c*X1)+d
        FXmin = (float(a*((X2)**3)))+(float(b*((X2)**2)))+(c*X1)+d
        X3 = X1-((FX1)*(X1-(X2)))/((FX1)-(FXmin))
        FXplus = (float(a*((X3)**3)))+(float(b*((X3)**2)))+(c*X3)+d
        if FXplus < 0:
            error = FXplus*(-1)
        else:
            error = FXplus
        if error > 0.0001:
            X2 = X1
            X1 = X3
        else:
            print("Selesai")
        if iterasi > 500:
            print("Angka Tak Hingga")
            break
        print(iterasi, "|", FX1, "|", FXmin, "|", X3, "|", FXplus, "|", error)
    print("Jumlah Iterasi = ", iterasi)
    print("Akar Persamaan = ", X3)
    print("Toleransi Error = ", error)
    
##### Metode Iterasi #####
def Iterasi (X1, a, b, c, d):
    X1 = X1
    a = a
    b = b
    c = c
    d = d
    error = 1
    iterasi = 0
    while (error > 0.0001):
        iterasi +=1
        Fxn = (float(a*((X1)**3)))+(float(b*((X1)**2)))+(c*X1)+d
        X2 = ((-a*(X1**2))+((b*3)*X1)+(-c))**(0.333334)
        Ea = (((X2-X1)/(X2))*100)
        if Ea < error:
            X1 = X2
            if Ea > 0:
                error = Ea
            else:
                error = Ea*(-1)
        else:
            error = Ea
        if iterasi > 100:
            print("Angka Tak Hingga")
            break
        print(iterasi, "|", X1, "|", X2, "|", Ea, "|", error)
    print("Jumlah Iterasi = ", iterasi)
    print("Akar Persamaan = ", X2)
    print("Toleransi Error = ", error)
    
##### MODUL 3 #####
#METODE ELEMINASI GAUSS
#Diketahui
#8a + 9b + 2c + 1d = 2.064
#2a + 7b + 1c + 3d = 0.664
#5a + 8b + 3c + 4d = 4.064
#1a + 6b + 5c + 2d = 2.064
import numpy as np
def Cal_LU_pivot(D, g):
    A = np.array((D), dtype=float)
    f = np.array((g), dtype=float)
    n = len(f)
    for i in range(0, n - 1):  # Looping untuk kolom matriks

        if np.abs(A[i, i]) == 0:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]] = A[[k, i]]  # Tukar antara baris i dan k
                    f[[i, k]] = f[[k, i]]
                    break

        for j in range(i + 1, n):  # Ulangi baris di bawah diagonal untuk setiap kolom
            m = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m * A[i, :]
            f[j] = f[j] - m * f[i]
    return A, f

#METODE ELEMINASI GAUSS JORDAN
#Diketahui
#8a + 9b + 2c + 1d = 2.064
#2a + 7b + 1c + 3d = 0.664
#5a + 8b + 3c + 4d = 4.064
#1a + 6b + 5c + 2d = 2.064
import numpy as np
import sys
def GaussJordan(a,n):
    #Step1 ===> Looping untuk pengolahan metode Gauss Jordan
    print('==============Mulai Iterasi===============')
    for i in range(n):
        if a[i][i]==0:
            sys.exit('Dibagi dengan angka nol (proses tidak dapat dilanjutkan)')
        for j in range(n):
            if i !=j:
                ratio=a[j][i]/a[i][i]
                #print('posisi nol di:[',j,i,']', 'nilai rasio:',ratio)
                
                for k in range (n+1):
                    a[j,k]=a[j][k]-ratio*a[i][k]
                print(a)
                print(f'============================================')
    # Step 2 ====> Membuat semua variabel(x,y,z,...)==1
    ax=np.zeros((n,n+1))
    for i in range(n):
        for j in range(n+1):
            ax[i,j]=a[i][j]/a[i][i]
    print('===================Akhir Iterasi============')
    return ax


#METODE ITERASI GAUSS SIEDEL
import numpy as np 
from scipy.linalg import solve 
def GaussSeidel (A, b, x, n):
    L = np.tril(A)
    U = A -L
    for i in range(n):
        x=np.dot(np.linalg.inv(L), b - np.dot(U,x))
        print (f'Iterasi Ke-{str(i+1).zfill(3)}'),
        print (x)
    return x

#METODE ITERASI JACOBI
import numpy as np
from scipy.linalg import solve
def jacobi(A,b,x,n):
    D = np.diag(A)
    R = A-np.diagflat(D)
    for i in range(n):
        x = (b-np.dot(R,x))/D
        print (f'Iterasi Ke-{str(i+1).zfill(3)}'),
        print (x)
    return x


##### MODUL 4 #####
import numpy as np
import matplotlib.pyplot as plt
#Trapesium 1 Pias
def trapesium_1pias():
    x = np.linspace(-10,10,100)
    y = 3*(x**3) + 3*(x**2)
    plt.plot(x,y)
    x1 = 0
    x2 = 1
    fx1 = 3*x1**3
    fx2 = 3*x2**2
    plt.fill_between([x1, x2], [fx1, fx2])
    plt.xlim([-15, 15]); plt.ylim([-2000,4000]);
    plt.title('Trapesium 1 Pias')
    plt.savefig('image\Trapesium1Pias.png')
    L = 0.5*(fx2 + fx1)*(x2 - x1)
    print("Luas dengan metode trapesium 1 pias:", L)
#Trapesium Banyak Pias
def trapesium_banyakpias(f,a,b,N):
    x = np.linspace(a,b,N+1)
    y = f(x)
    y_kanan = y[1:]
    y_kiri = y[:-1]
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_kanan + y_kiri)
    return T
f = lambda x : 3*(x**3) + 3*(x**2) + 3
a = 0
b = 10
N = 5
x = np.linspace(a,b,N+1)
y = f(x)
X = np.linspace(a,b+1,N)
Y = f(X)
#Metode Simpson 1/3
def simpson1per3(x0,xn,n):
    f = lambda x: 3*(x**3) + 3*(x**2) + 3
    h = (xn - x0) / n
    integral = f(x0) + f(xn)
    for i in range(1,n):
        k = x0 + i*h
        if i%2 == 0:
            integral = integral + 2 * f(k)
        else:
            integral = integral + 4 * f(k)
    integral = integral * h / 3
    return integral
#Simpson 3/8
def simpson3per8(x0,xn,n):
    f = lambda x: 3*(x**3)+3*(x**2) +3
    h = (xn - x0) / n
    integral = f(x0) + f(xn)
    for i in range(1,n):
        k = x0 + i*h
        if i%2 == 0:
            integral = integral + 3 * f(k)
        else:
            integral = integral + 3 * f(k)
    integral = integral * 3 * h / 8
    return integral
# Metode Simpson 1/3 Dua Pias
def f(x):
    #persamaan : 3x^3+4x^2+8
    return 3*x**3+4*x**2+8
def simpson1per3duapias(x0,xn,n):
    h = (xn - x0) / n
    integral = f(x0) + f(xn)
    for i in range(1,n):
        k = x0 + i*h 
        if i%2 == 0:
            integral = integral + 2 * f(k)
        else:
            integral = integral + 4 * f(k)
    integral = integral * h/3
    return print("Hasil nilai integral Metode Simpson 1/3 adalah",+integral)

##### MODUL 5 #####
#=== Modul 5 Persamaan Diferensial Biasa ===#
#==== Metode Euler ===#
def Euler(h, x0, xn, y0, a, b, c, d):
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    plt.style.use('seaborn-poster')
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
    h = h
    x0 = x0
    xn = xn
    x = np.arange(x0, xn + h, h)
    y0 = y0
    a = a
    b = b
    c = c
    d = d
    G = (a*(x**3))+(b*(x**2))+(c*x)+d
    f = lambda x, y: (a*(x**3))+(b*(x**2))+(c*x)+d
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0, len(x) - 1):
        y[i + 1] = y[i] + h*f(x[i], y[i])
    Galat = G-y
    print (Galat)
    Judul = ("Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Euler")
    plt.figure(figsize = (12, 12))
    plt.plot(x, y, 'b--', label='Hasil Pendekatan')
    plt.plot(x, G, '-g', label='Hasil Analitik')
    plt.title(Judul) # Judul plot
    plt.xlabel('x') # Label sumbu x
    plt.ylabel('F(x)') # Label sumbu y
    plt.grid() # Menampilkan grid
    plt.legend(loc='lower right')
    plt.savefig('Image\euler.png')
    
#==== Metode Heun ===#
def Heun(h, x0, xn, y0, a, b, c, d):
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    plt.style.use('seaborn-poster')
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
    h = h
    x0 = x0
    xn = xn
    y0 = y0
    a = a
    b = b
    c = c
    d = d
    x = np.arange(x0, xn + h, h)
    G = (a*(x**3))+(b*(x**2))+(c*x)+d
    f = lambda x, y: (a*(x**3))+(b*(x**2))+(c*x)+d
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(0, len(x) - 1):
        k1 = f(x[i], y[i])
        k2 = f((x[i]+h), (y[i]+(h*k1)))
        y[i+1] = y[i]+(0.5*h*(k1+k2))
    Galat = G-y
    print (Galat)
    Judul = ("Grafik Pendekatan Persamaan Differensial Biasa Dengan Metode Heun")
    plt.figure(figsize = (12, 12))
    plt.plot(x, y, 'b--', label='Hasil Pendekatan')
    plt.plot(x, G, '-g', label='Hasil Analitik')
    plt.title(Judul) # Judul plot
    plt.xlabel('x') # Label sumbu x
    plt.ylabel('F(x)') # Label sumbu y
    plt.grid() # Menampilkan grid
    plt.legend(loc='lower right')
    plt.savefig('Image\heun.png')
    
#########################################################################################
######################################### BATAS #########################################
#########################################################################################

##### Output #####
print("Kode Persamaan : \n",
     "1. Metode Setengah Interval \n",
     "2. Metode Interpolasi Linear \n",
     "3. Metode Newton-Rhapson \n",
     "4. Metode Secant \n",
     "5. Metode Iterasi \n",
     "6. Metode Gauss \n",
     "7. Metode Gauss Jordan \n",
     "8. Metode Gauss Seidel \n",
     "9. Metode Jacobi \n",
     "10 Metode Trapesium 1 Pias \n",
     "11. Metode Trapesium Banyak Pias \n",
     "12. Metode Simpson 1/3 \n",
     "13. Metode Simpson 3/8 \n",
     "14. Metode Simpson 1/3 Dua Pias \n",
     "15. Euler \n",
     "16. Heun \n")
setting = int(input("Masukkan Kode Persamaan :"))
##### MODUL 2 #####
if (setting == 1):
    X1 = float(input("Masukkan Nilai Pertama : "))
    X2 = float(input("Masukkan Nilai Kedua : "))
    a = float(input("Masukkan Nilai a : "))
    b = float(input("Masukkan Nilai b : "))
    c = float(input("Masukkan Nilai c : "))
    d = float(input("Masukkan Nilai d : "))
    X = SetengahInterval(X1, X2, a, b, c, d)
    print(X)
elif (setting == 2):
    X1 = float(input("Masukkan Nilai X1 untuk Interpolasi Linear : "))
    a = float(input("Masukkan Nilai a : "))
    b = float(input("Masukkan Nilai b : "))
    c = float(input("Masukkan Nilai c : "))
    d = float(input("Masukkan Nilai d : "))
    X2 = InterpolasiLinear (X1, a, b, c, d)
    print(X2)
elif (setting == 3):
    X1 = float(input("Masukkan Nilai X1 untuk Newton-Rhapson : "))
    a = float(input("Masukkan Nilai a : "))
    b = float(input("Masukkan Nilai b : "))
    c = float(input("Masukkan Nilai c : "))
    d = float(input("Masukkan Nilai d : "))
    X2 = NewtonRhapson(X1, a, b, c, d)
    print(X2)
elif (setting == 4):
    X1 = float(input("Masukkan Nilai X1 untuk Secant : "))
    a = float(input("Masukkan Nilai a : "))
    b = float(input("Masukkan Nilai b : "))
    c = float(input("Masukkan Nilai c : "))
    d = float(input("Masukkan Nilai d : "))
    X2 = Secant(X1, a, b, c, d)
    print(X2)
elif (setting == 5):
    X1 = float(input("Masukkan Nilai X1 untuk Secant : "))
    a = float(input("Masukkan Nilai a : "))
    b = float(input("Masukkan Nilai b : "))
    c = float(input("Masukkan Nilai c : "))
    d = float(input("Masukkan Nilai d : "))
    X2 = Iterasi(X1, a, b, c, d)
    print(X2)
    
##### MODUL 3 #####  
elif (setting == 6):
    A = np.array([[8,9,2,1], [2,7,1,3],[5,8,3,4], [1,6,5,2]], dtype=float)
    f = np.array([2.064, 0.664, 4.064, 2.064])
    print('A = \n%s dan f = %s' % (A, f))
    B, g = Cal_LU_pivot(A, f)
    print (B,g)
    x = np.linalg.solve(A, f)
    print('Hasil perhitungan Metode Gauss adalah x = \n %s' % x)
elif (setting == 7):
    m = np.array([[8,9,2,1,2.064], 
                  [2,7,1,3,0.664], 
                  [5,8,3,4,4.064], 
                  [1,6,5,2,2.064]],dtype=float)
    n = int(input("Masukkan Nilai n Gauss Jordan (Bilangan Bulat) :"))
    # Menampilkan matriks awal
    print('Matriks Persamaan')
    print(m)
    # Menampilkan Hasil
    m = GaussJordan(m,n)
    print(f"""Hasil perhitungan metode Gauss Jordan didapatkan matriks: 
    {m}""")
elif (setting == 8):
  #Masukan
    A= np.array([[8,9,2,1], 
                   [2,7,1,3],
                   [5,8,3,4], 
                   [1,6,5,2]],dtype=float)
    b = [2.064, 0.664, 4.064, 2.064]
    x = np.zeros_like(b)
    n = int(input("Masukkan Nilai n Gauss Seidel (Bilangan Bulat) :"))
    H = GaussSeidel(A, b, x, n)
    K = solve(A,b)
    print(f'Hasil pengerjaan menggunakan Gauss Seidel didapatkan nilai tiap variabel {H}')
    print(f'Variabel matriks menggunakan SciPy {K}')
elif (setting == 9):
    A= np.array([[8,9,2,1], 
                   [2,7,1,3],
                   [5,8,3,4], 
                   [1,6,5,2]],dtype=float)
    b=[2.064, 0.664, 4.064, 2.064]
    x = [1.0,1.0,1.0,1.0]
    n = int(input("Masukkan Nilai n Jacobi (Bilangan Bulat) :"))
    J = jacobi(A,b,x,n)
    O = solve(A,b)
    print(f'Hasil pengerjaan menggunakan Jacobi didapatkan nilai tiap variabel {J}')
    print(f'Variabel matriks menggunakan SciPy {O}')

#### MODUL 4 #####  
elif (setting == 10):
    Cel = trapesium_1pias()
elif (setting == 11):
    plt.plot(X,Y)
    for i in range(N):
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys, 'b', edgecolor='b',alpha=0.2)
    plt.title('Trapesium banyak pias, N = {}'.format(N))
    L = trapesium_banyakpias(f,a,b,N)
    print(L)
elif (setting == 12):
    x1 = float(input("Batas bawah (a): "))
    x2 = float(input("Batas bawah (b): ")) #Isi dengan nilai 10 karna sesuai dalam tugas h+6
    hasil = simpson1per3(x1, x2, 3)
    print("Nilai integral metode Simpson 1/3:", hasil)
elif (setting == 13):
    x1 = float(input("Batas bawah (x1): "))
    x2 = float(input("Batas atas (x2): "))  #Isi dengan nilai 10 karna sesuai dalam tugas h+6
    hasil = simpson3per8(x1, x2, 3)
    print("Nilai integral metode Simpson 3/8:", hasil)
elif (setting == 14):
    #x0(batas bawah)=0 dan xn(batas atas)=10, sesuai ketentuan
    hasil = simpson1per3duapias(0, 10, 2)
    print("Nilai integral metode Simpson 3/8:", hasil)

#### MODUL 5 #####     
elif (setting == 15):
    h = float(input("Masukkan nilai h: "))
    x0 = float(input("Masukkan nilai x awal x0: "))
    xn = float(input("Masukkan nilai x akhir xn: "))
    y0 = float(input("Masukkan nilai y awal y0: "))
    a = float(input("Masukkan Nilai a Euler: "))
    b = float(input("Masukkan Nilai b Euler: "))
    c = float(input("Masukkan Nilai c Euler: "))
    d = float(input("Masukkan Nilai d Euler: "))
    X = Euler(h, x0, xn, y0, a, b, c, d)
    print(X)
elif (setting == 16):
    h = float(input("Masukkan nilai h: "))
    x0 = float(input("Masukkan nilai x awal x0: "))
    xn = float(input("Masukkan nilai x akhir xn: "))
    y0 = float(input("Masukkan nilai y awal y0: "))
    a = float(input("Masukkan Nilai a Heun: "))
    b = float(input("Masukkan Nilai b Heun: "))
    c = float(input("Masukkan Nilai c Heun: "))
    d = float(input("Masukkan Nilai d Heun: "))
    X = Heun(h, x0, xn, y0, a, b, c, d)
    print(X)

else:
    print("Periksa Kembali Kode yang Diminta!")


# ### 

# In[ ]:




