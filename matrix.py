import pandas as pd
import numpy as np
import math

def treino():
    thetas = np.random.random(785)

    dadosTreino = pd.read_csv('train.csv').values
    labels = dadosTreino[:,0]
    dadosTreino = dadosTreino[:,1:]

    labels = to_matrix( labels )
    dadosTreino = dadosTreino/255
    
    dadosTreinoT = dadosTreino.T

    #theta = inv(xT * x) * xT * y
    aux = np.dot( dadosTreinoT, dadosTreino )
    aux = (aux + 0.0000001*np.random.rand(784, 784)).astype(float)
    aux_inv = np.linalg.inv( aux )
    aux = np.dot( aux_inv, dadosTreinoT )
    aux = np.dot( aux, labels )
    thetas = aux

    return thetas


def teste( thetas ):

    saida = open("saida_matrix.csv", 'w'); 
    saida.write("ImageId,Label\n")


    dadosTeste = pd.read_csv('test.csv').values
    dadosTeste = dadosTeste/255

    for i in range( len( dadosTeste ) ):
        d = dadosTeste[i]
        aux = np.argmax( np.dot(d, thetas) )

        result = int( aux )
        if result > 9:
            result = 9
        if result < 0:
            result = 0
        saida.write(str(i+1)+","+str(int(result))+"\n")


def to_matrix( labels ):

    matrix_labels = np.zeros((labels.shape[0],10))

    i=0    
    for label in labels:
        matrix_labels[i][label] = 1
        i+=1
        
    print(matrix_labels)

    return matrix_labels


def main():
 
    thetas = treino()
    teste( thetas )

main()