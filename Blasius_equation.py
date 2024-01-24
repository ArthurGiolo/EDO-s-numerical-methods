import matplotlib.pyplot as plt
import numpy as np

# Sistema de equações
def f_linha(u):
    return u

def u_linha(v):
    return v

def v_linha(f,v):
    return -0.5*f*v

# Definindo função para NR
def NR(f,eta):

    # Fazer Blasius 
    a   = Blasius(f)
    F_a = a[1][-1] - 1

    # Fazer Blasius com um difetencial
    b   = Blasius(f + eta)
    F_b = b[1][-1] - 1

    return f - (F_a*eta)/(F_b - F_a), a[0],a[1],a[2],a[3]

# Definindo RK6
def RK6(f,u,v,h):
    
    # Calculo de C1
    k1 = f_linha(u)
    l1 = u_linha(v)
    m1 = v_linha(f,v)

    # Calculo de C2
    k2 = f_linha(u + 0.25*h*l1)
    l2 = u_linha(v + 0.25*h*m1)
    m2 = v_linha(f + 0.25*h*k1, v + 0.25*h*m1)

    # Calculo de C3
    k3 = f_linha(u + 0.125*h*l1 + 0.125*h*l2)
    l3 = u_linha(v + 0.125*h*m1 + 0.125*h*m2)
    m3 = v_linha(f + 0.125*h*k1 + 0.125*h*k2, v + 0.125*h*m1 + 0.125*h*m2)
    
    # Calculo de C4
    k4 = f_linha(u - 0.5*h*l2 + h*l3)
    l4 = u_linha(v - 0.5*h*m2 + h*m3)
    m4 = v_linha(f - 0.5*h*k2 + h*k3,v - 0.5*h*l2 + h*l3)

    # Calculo de C5
    k5 = f_linha(u + 0.1875*h*l1 + 1.125*h*l4)
    l5 = u_linha(v + 0.1875*h*m1 + 1.125*h*m4)
    m5 = v_linha(f + 0.1875*h*k1 + 1.125*h*k4, v + 0.1875*h*m1 + 1.125*h*m4)

    # Calculo de C6
    k6 = f_linha(u - (1/7)*(3*h*l1 - 2*h*l2 - 12*h*l3 + 12*h*l4 - 8*h*l5))
    l6 = u_linha(v - (1/7)*(3*h*m1 - 2*h*m2 - 12*h*m3 + 12*h*m4 - 8*h*m5))
    m6 = v_linha(f - (1/7)*(3*h*k1 - 2*h*k2 - 12*h*k3 + 12*h*k4 - 8*h*k5),v - (1/7)*(3*h*m1 - 2*h*m2 - 12*h*m3 + 12*h*m4 - 8*h*m5))
    
    f  = f + (1/90)*h*(7*k1 + 32*k2 + 12*k4 + 32*k5 + 7*k6)
    u  = u + (1/90)*h*(7*l1 + 32*l2 + 12*l4 + 32*l5 + 7*l6)
    v  = v + (1/90)*h*(7*m1 + 32*m2 + 12*m4 + 32*m5 + 7*m6)

    return f,u,v

# Definindo o passo adaptativo
def Adap(f,u,v,h):

    # definindo tolerância para o passo adaptativo
    tol = 1e-5

    # Código fica rodando até achar um valor onde o passo h seja aceitável
    i = 1
    while i != 0:
        
        # Primeiro RK6
        val1 = RK6(f,u,v,h)

        # Segundo RK6
        val2 = RK6(f,u,v,0.5*h)

        # Caluclando o erro
        eta  = (val1[0] - val2[0])/(2**7 - 1)   

        if eta > tol:
            h = 0.8*h*(tol/eta)**(1/6)
        elif eta <= 0.5*tol:
            h = 3*h
        elif eta > 0.5*tol and eta <= tol:
            return val1[0], val1[1], val1[2], h

# Função para RK6 com passo adaptrativo
def Blasius(c):
    h = 0.1
    n = [0]
    f = [0]
    u = [0]
    v = [c]
    while n[-1] <= 7:
        # Fazendo o passo adaptativo
        a   = Adap(f[-1],u[-1],v[-1],h)

        # Andar o passo:
        h   = a[-1]
        val = n[-1] + h

        # Adicionar valores calculados para as variáveis
        f.append(a[0])
        u.append(a[1])
        v.append(a[2])
        n.append(val)
    
    return f,u,v,n

f = 0.1
b = 2
while round(b,7) != 1:
    n = NR(f,1e-4)
    f = n[0]
    b = n[2][-1]

print(f)

plt.plot(n[2], n[-1])
plt.title('Solução numérica da equação de Blasius')
plt.xlabel(u'f\'')
plt.ylabel(u"\u03B7")
plt.show()