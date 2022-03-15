## Objetivo

O conjunto de dados MUTAG é uma coleção de compostos nitroaromáticos e o objetivo deste trabalho é representar esses compostos como grafos e utilizar uma rede neural de grafos para prever a mutagenicidade na Salmonella Typhimurium.

## Estrutura dos dados

As entradas do algoritmo são compostos químicos representados através de grafos, os vértices representam os átomos do composto e são categorizados de acordo com o seu tipo de átomo (representado utilizando `one_hot_encoding`), enquanto as arestas entre os vértices representam as ligações entre dois átomos.

O conjunto de dados inclue 188 amostras de compostos com 7 tipos de átomos diferentes.


## Estrutura do projeto

- `mutag/`: Está a biblioteca utilizada para carregar o dataset, definir a arquitetura do modelo e o código utilizado para treino e teste.
- `notebooks/`: Estão todos os notebooks utilizados para estudo e experimentos.
