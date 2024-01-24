import matplotlib.pyplot as plt
import numpy as np

# Valor inicial
alpha = 50

# Variação temporal
dt = 0.0001
t = np.arange(0, 30.0 + dt, dt)

# EDO's
def r_ponto(u):
    return u

def u_ponto(w,r,teta):
    return r*(w**2) + np.cos(teta) - alpha*(r - 1)

def teta_ponto(w):
    return w

def w_ponto(u,w,r,teta):
    return (-2*u*w - np.sin(teta))/r


# Variáveis
r             = np.zeros(len(t))
r[0]          = 1
teta          = np.zeros(len(t))
teta[0]       = 0.9*np.pi
u             = np.zeros(len(t))
u[0]          = 0
w             = np.zeros(len(t))
w[0]          = 0.0*np.pi

# Salvar a função pra depois 
d1            = np.zeros(len(t))
d1[0]         = r_ponto(u[0])
d2            = np.zeros(len(t))
d2[0]         = u_ponto(w[0], r[0], teta[0])
d3            = np.zeros(len(t))
d3[0]         = teta_ponto(w[0])
d4            = np.zeros(len(t))
d4[0]         = w_ponto(u[0],w[0],r[0],teta[0])

# RK4 para conseguir os pontos
for i in range(4):

    # Primeiro ponto do RK
    d1[i] = r_ponto(u[i])
    d2[i] = u_ponto(w[i], r[i], teta[i])
    d3[i] = teta_ponto(w[i])
    d4[i] = w_ponto(u[i],w[i],r[i],teta[i])

    # Segundo ponto do RK
    k2 = r_ponto(u[i] + 0.5*dt*d2[i])
    l2 = u_ponto(w[i] + 0.5*dt*d4[i], r[i] + 0.5*dt*d1[i], teta[i] + 0.5*dt*d3[i])
    m2 = teta_ponto(w[i] + 0.5*dt*d4[i])
    n2 = w_ponto(u[i] + 0.5*dt*d2[i], w[i] + 0.5*dt*d4[i],r[i] + 0.5*dt*d1[i], teta[i] + 0.5*dt*d3[i])

    # Terceiro ponto do RK
    k3 = r_ponto(u[i] + 0.5*dt*l2)
    l3 = u_ponto(w[i] + 0.5*dt*n2, r[i] + 0.5*dt*k2, teta[i] + 0.5*dt*m2)
    m3 = teta_ponto(w[i] + 0.5*dt*n2)
    n3 = w_ponto(u[i] + 0.5*dt*l2, w[i] + 0.5*dt*n2,r[i] + 0.5*dt*k2, teta[i] + 0.5*dt*m2)

    # Quarto ponto do RK
    k4 = r_ponto(u[i] + dt*l3)
    l4 = u_ponto(w[i] + dt*n3, r[i] + dt*k3, teta[i] + dt*m3)
    m4 = teta_ponto(w[i] + dt*n3)
    n4 = w_ponto(u[i] + dt*l3, w[i] + dt*n3,r[i] + dt*k3, teta[i] + dt*m3)

    r[i+1]    = r[i]    + (1/6)*dt*(d1[i] + 2*k2 + 2*k3 + k4)
    u[i+1]    = u[i]    + (1/6)*dt*(d2[i] + 2*l2 + 2*l3 + l4)
    teta[i+1] = teta[i] + (1/6)*dt*(d3[i] + 2*m2 + 2*m3 + m4)
    w[i+1]    = w[i]    + (1/6)*dt*(d4[i] + 2*n2 + 2*n3 + n4)


# Adams-Bashforth-Moulton 
for i in range(3, len(t) - 1):
    ## Adams-Bashforth
    r[i+1]    = r[i]    + (1/24)*dt*(55*d1[i] - 59*d1[i-1] + 37*d1[i-2] - 9*d1[i-3])
    u[i+1]    = u[i]    + (1/24)*dt*(55*d2[i] - 59*d2[i-1] + 37*d2[i-2] - 9*d2[i-3])
    teta[i+1] = teta[i] + (1/24)*dt*(55*d3[i] - 59*d3[i-1] + 37*d3[i-2] - 9*d3[i-3])
    w[i+1]    = w[i]    + (1/24)*dt*(55*d4[i] - 59*d4[i-1] + 37*d4[i-2] - 9*d4[i-3])
    
    ## Calculando novos pontos das funções
    d1[i+1] = r_ponto(u[i+1])
    d2[i+1] = u_ponto(w[i+1], r[i+1], teta[i+1])
    d3[i+1] = teta_ponto(w[i+1])
    d4[i+1] = w_ponto(u[i+1],w[i+1],r[i+1],teta[i+1])
    
    ## Adams-Moulton
    r[i+1]    = r[i]    + (1/24)*dt*(9*d1[i+1] + 19*d1[i] - 5*d1[i-1] + d1[i-2])
    u[i+1]    = u[i]    + (1/24)*dt*(9*d2[i+1] + 19*d2[i] - 5*d2[i-1] + d2[i-2])
    teta[i+1] = teta[i] + (1/24)*dt*(9*d3[i+1] + 19*d3[i] - 5*d3[i-1] + d3[i-2])
    w[i+1]    = w[i]    + (1/24)*dt*(9*d4[i+1] + 19*d4[i] - 5*d4[i-1] + d4[i-2])

    ## Corrigindo os pontos das funções
    d1[i+1] = r_ponto(u[i+1])
    d2[i+1] = u_ponto(w[i+1], r[i+1], teta[i+1])
    d3[i+1] = teta_ponto(w[i+1])
    d4[i+1] = w_ponto(u[i+1],w[i+1],r[i+1],teta[i+1])



plt.plot(t,teta)
plt.title("Problema pendulo - Ângulo")
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (rad)')
plt.legend()
plt.show()

plt.plot(t,w)
plt.title("Problema pendulo - Velocidade angular")
plt.xlabel('Tempo (s)')
plt.ylabel(' Velocidade angular (rad/s)')
plt.show()

plt.plot(t,u)
plt.title("Problema pendulo - Taxa de deformação")
plt.xlabel('Tempo (s)')
plt.ylabel('Taxa de deformação (m/s)')
plt.show()

plt.plot(t,r)
plt.title("Problema pendulo - Raio")
plt.xlabel('Tempo (s)')
plt.ylabel('Raio (m)')
plt.show()


