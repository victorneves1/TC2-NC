# TC2-NC

## Trabalho computacional sobre algoritmos evolutivos para solução de sistemas de equações não lineares 

Alunos: João Gobeti Calenzani e Victor Nascimento Neves


## Descrição

Esse trabalho computacional tem como objetivo a implementação de algoritmos evolutivos para a solução de sistemas de equações não lineares. Mais informações no relatŕio em PDF no repositório (TC2_NC - Relatório.pdf).

## Instalação

Requer Python 3, virtualenv e Jupyter Notebook.
Desenvolvido e testado no Python 3.10.0, Ubuntu 22.04.

Os módulos necessários estão listados no arquivo requirements.txt. Para instalar com virtualenv, execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Os experimentos estão contido no notebook Jupyter `tc2.ipynb`. Para executar, execute:

```bash
jupyter-notebook tc2.ipynb
```

### Detalhes do Notebook

- 1ª célula Python do notebook: define os sistemas de equações não lineares a serem resolvidos (P1 e P2)
- 2ª, 3ª e 4ª células Python do notebook: solução do P1 usando GA
- 5ª, 6ª e 7ª células Python do notebook: solução do P2 usando GA
- 8ª, 9ª e 10ª células Python do notebook: solução do P1 usando ES
- 11ª, 12ª e 13ª células Python do notebook: solução do P2 usando ES


### Código Fonte
 
A pasta `src/` contém as implementações dos algoritmos genéticos e estratégias evolutivas. Os arquivos são:
- ga.py
- es.py