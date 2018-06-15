import pandas as pd
import numpy as np

dados_treinamento = pd.read_csv('train.csv').values

y = dados_treinamento[:, 0]
X = dados_treinamento[:, 1:]

theta = np.random.random(785)

xl = np.hstack((np.ones(42000).reshape(-1,1),X))
diferenca = xl.dot(theta) - y

erro = np.linalg.norm(diferenca)

derivada = (2.0*(diferenca.T)).dot(xl)
print(derivada)

alfa = 1e-5
while(1):
	derivada = (2.0*(diferenca.T)).dot(xl)
	derivada_alfa = derivada*alfa
	theta = theta - derivada_alfa

	diferenca = xl.dot(theta) - y

	erro = np.linalg.norm(diferenca)


	print(erro)



