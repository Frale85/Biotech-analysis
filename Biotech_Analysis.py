#In questo blocco, vengono importate le librerie.
#Le principali librerie sono NumPy, 
#Matplotlib, SciPy e
#SALib per l'analisi di sensitività. 
#Viene anche importata la libreria warnings per gestire i messaggi di avviso.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings


#definisco i parametri del modello SEIR,
#il numero totale di individui (N), i tassi di contatto (beta), di guarigione (gamma)
#creiamo dei punti temporali (t) per la simulazione

N = 350.0  
I0, R0 = 1.0, 0.0  #condizioni iniziali
S0 = N - I0 - R0  
beta, gamma, sigma = 0.4, 0.1, 0.05  
Tmax = 160 
Nt = 160
t = np.linspace(0, Tmax, Nt + 1)

#calcola le derivate di un sistema di equazioni differenziali ordinarie (ODE)
#resttuisce un array NumPy contenente le derivate calcolate nelle quattro equazioni.

def derivative(X, t):
    S, E, I, R = X
    dotS = -beta * S * I / N
    dotE = beta * S * I / N - sigma * E
    dotI = sigma * E - gamma * I
    dotR = gamma * I
    return np.array([dotS, dotE, dotI, dotR])

#Definizmo le condizioni iniziali (X0) e risolte le equazioni differenziali
#del modello SEIR usando la funzione odeinty. 
#Le curve temporali vengono estratte e salvate in S, E, I e R. 
#calcoliamo anche il valore di soglia Seuil.

X0 = S0, I0, R0, 0  # Initial condition vector
res = integrate.odeint(derivative, X0, t)
S, E, I, R = res.T
Seuil = 1 - 1 / (beta / gamma)

#i risultati della simulazione vengono visualizzati tramite un grafico.
#che mostra l'andamento delle quattro variabili del modello nel tempo.

plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, S, 'orange', label='Susceptible')
plt.plot(t, E, 'm', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered with immunity')
plt.xlabel('Time t, [days]')
plt.ylabel('Number of individuals')
plt.ylim([0, N])
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings

#definiamo una classe SEIRModel che contine il modello SEIR.
#e'presente costruttore __init__ e un metodo derivative che rappresenta
#le equazioni differenziali del modello, ma questa volta la funzione derivative
#prende self come primo argomento

# Definizione del modello SEIR
class SEIRModel:
    def __init__(self, N):
        self.N = N

    def derivative(self, X, t, beta, gamma, sigma):
        S, E, I, R = X
        dotS = -beta * S * I / self.N
        dotE = beta * S * I / self.N - sigma * E
        dotI = sigma * E - gamma * I
        dotR = gamma * I
        return [dotS, dotE, dotI, dotR]

# Parametri del modello
N = 350.0
beta = 0.4
gamma = 0.1
sigma = 0.05
Tmax = 160
Nt = 160
t = np.linspace(0, Tmax, Nt + 1)

# Creazione dell'istanza del modello SEIR
model = SEIRModel(N)

#definiamo tre diverse configurazioni 
#di condizioni iniziali per il modello SEIR. Ogni configurazione 
#è rappresentata da un set di valori iniziali 
#per Suscettibili (S), Infetti (I) e Guariti (R).

# Configurazione 1 delle condizioni iniziali
I0_1, R0_1 = 1.0, 0.0
S0_1 = N - I0_1 - R0_1
X0_1 = S0_1, I0_1, R0_1, 0

# Configurazione 2 delle condizioni iniziali
I0_2, R0_2 = 5.0, 2.0
S0_2 = N - I0_2 - R0_2
X0_2 = S0_2, I0_2, R0_2, 0

# Configurazione 3 delle condizioni iniziali
I0_3, R0_3 = 10.0, 5.0
S0_3 = N - I0_3 - R0_3
X0_3 = S0_3, I0_3, R0_3, 0


#eseguiamo le simulazioni della dinamica del modello SEIR
#per le tre diverse configurazioni iniziali. Le funzioni odeint di SciPy
#vengono utilizzate per risolvere le equazioni differenziali del modello.
# Simulazione della dinamica del modello per le tre configurazioni iniziali
res_1 = integrate.odeint(model.derivative, X0_1, t, args=(beta, gamma, sigma))
res_2 = integrate.odeint(model.derivative, X0_2, t, args=(beta, gamma, sigma))
res_3 = integrate.odeint(model.derivative, X0_3, t, args=(beta, gamma, sigma))

#le curve temporali per le variabili Suscettibili (S), Esposti (E),
#Infetti (I) e Guariti (R) vengono estratte dalle simulazioni per le tre diverse configurazioni.
# Estrazione delle curve per le tre configurazioni iniziali
S_1, E_1, I_1, R_1 = res_1.T
S_2, E_2, I_2, R_2 = res_2.T
S_3, E_3, I_3, R_3 = res_3.T

#in questo blocco vengono visualizzate le curve temporali per le variabili
#Suscettibili (S), Infetti (I) e Guariti (R) per le tre diverse configurazioni 

plt.figure()
plt.grid()
plt.title("Dinamica del modello SEIR per diverse configurazioni iniziali")
plt.plot(t, S_1, 'orange', label='Susceptible (Configurazione 1)')
plt.plot(t, I_1, 'r', label='Infected (Configurazione 1)')
plt.plot(t, R_1, 'g', label='Recovered (Configurazione 1)')

plt.plot(t, S_2, 'blue', label='Susceptible (Configurazione 2)')
plt.plot(t, I_2, 'purple', label='Infected (Configurazione 2)')
plt.plot(t, R_2, 'cyan', label='Recovered (Configurazione 2)')

plt.plot(t, S_3, 'brown', label='Susceptible (Configurazione 3)')
plt.plot(t, I_3, 'pink', label='Infected (Configurazione 3)')
plt.plot(t, R_3, 'yellow', label='Recovered (Configurazione 3)')

plt.xlabel('Time t, [days]')
plt.ylabel('Number of individuals')
plt.ylim([0, N])
plt.legend()
plt.show()
