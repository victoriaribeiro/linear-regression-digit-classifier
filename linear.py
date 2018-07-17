import pandas as pd
import numpy as np
 
def treino():
	matriz = pd.read_csv('train.csv').values

	x = matriz[:, 1:]	
	x = x/255

	y = matriz[:, 0]
	
	theta = np.random.random(785)	
	
	xlinha = np.hstack((np.ones(42000).reshape(-1, 1), x))	
	
	erro = (xlinha.dot(theta) - y).T 	
	derivada = 2 * erro.dot(xlinha)
	
	alfa = 3e-7
	E = 1e-5
	while(1):
		erro = (xlinha.dot(theta) - y).T 
	
		derivada = alfa * (2 * erro.dot(xlinha))

		thetalinha = theta - derivada
		print (np.linalg.norm(erro))	
		print(np.linalg.norm(theta - thetalinha))
	
		if(np.linalg.norm(theta - thetalinha) < E):
			break
		theta = thetalinha

	return theta
	
	
def teste(theta):

	saida = open("saida.csv", 'w')

	dadosTeste = pd.read_csv('test.csv').values	
	dadosTeste = dadosTeste/255
	
	dadosTeste = np.hstack((np.ones(dadosTeste.shape[0]).reshape(-1, 1), dadosTeste))	
	result = dadosTeste.dot(theta)

	saida.write("ImageId,Label\n")
	for i in range(result.shape[0]):
		output = result[i]
		if(output>9):
			output = 9
		elif (output<0):
			output = 0
		saida.write(str(i+1)+","+str(int(output))+"\n")
	

def main():
	theta = treino()
	teste(theta)

main()
