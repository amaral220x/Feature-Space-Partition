# Feature-Space-Partition
A local-global classification scheme for Python, based on MATLAB version.
![](https://github.com/amaral220x/Feature-Space-Partition/blob/main/multipage-inf.gif)

## Como usar
Em primeiro momento, é necessário instalar as dependências do projeto. Para isso, execute o comando abaixo:
```bash
pip install -r requirements.txt
```
Ou
```bash
pip3 install -r requirements.txt
```

Após a instalação das dependências, use o arquivo `sample_test.py` para fazer uma pequena bateria de testes:
```bash 
python sample_test.py <nomedabase>.mat
```
Lembre-se que o dataset deve estar no formato `.mat` e deve estar na pasta `Datasets`.

## Sobre os arquivos
- `sample_test.py`: Arquivo de teste para o projeto.
- `fsp.py`: Feature Space Partition = Arquivo principal do projeto, onde o particionamento é feito.
- `d_cs.py`: Divergence Measure = Arquivo auxiliar para o cálculo da medida de divergência de CS.
- `pdc.py`: Probabilistic Draw Classifier = Classificador baseado no particionamento
- `IlustrationsPDF.ipynb`: Notebook que gera plots da segmentação do espaço após o FSP.
- `IlustrationsGIF.ipynb`: Notebook que gera um GIF com a segmentação do espaço após o FSP.

## Notebooks
O repositório conta com Notebooks que ilustram melhor o funcionamento das funções principais e também conta com as decisões na limpagem dos Datasets.



## Feito por
- [Gabriel Amaral](https://github.com/amaral220x)
- [Bruno Carmo](https://la.mathworks.com/matlabcentral/profile/authors/7908674)
- [Carolina Gil Marcelino](https://www.mathworks.com/matlabcentral/fileexchange/112245-feature-space-partition-fsp)
