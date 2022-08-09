# DroidAutoML

DroidAutoML é uma ferramenta de AutoML de domínio específico que abstrai a execução das etapas de limpeza de dados, engenharia de características, escolha de algoritmos e ajuste de hiper-parâmetros, identificando o  modelo que melhor se ajusta ao problema de classificação de malwares Android. 

A DroidAutoML também implementa níveis de personalização, transparência, interpretabilidade e depuração que não são comuns nas ferramentas de AutoML de proposito geral, que funcionam essencialmente como "caixas pretas" (e.g., apenas apresentação das métricas finais aos usuários).

## O pipeline de AutoML

![O pipeline da DroidAutoML](https://gcdnb.pbrd.co/images/ZLQfWKF12ZN5.png?o=1)

**Etapa 1**: o **pré-processamento dos dados** trata valores ruidosos com:
- valores faltantes NaN (ou preenchê-los dependendo do caso);
- valores nulos (ou preenchê-los dependendo do caso);
- características/colunas que contenham apenas valor "0" zero para todas as amostras.

**Etapa 2**: na **engenharia de características**, a principal tarefa é identificar aquelas características mais relevantes para o domínio do problema, utilizando métodos sofistidados de seleção de características; 

Métodos atualmente disponíveis:
- [SigPID](https://ieeexplore.ieee.org/document/7888730): especializado em permissões
- [RFG](https://www.mdpi.com/2079-9292/9/3/435): especializado em chamadas de API
- [JOWMDroid](https://www.sciencedirect.com/science/article/pii/S016740482030359X): diversos tipos de características

**Etapa 3**: na **seleção de modelos** a ferramenta recebe o dataset resultante da etapa anterior e aplica 3 algoritmos de aprendizado de máquina:
- [K-Nearest Neighbor(KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Random forest](https://en.wikipedia.org/wiki/Random_forest)
- [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)

**Etapa 4**: no **ajuste do modelo** os algoritmos da etapa 3 são otimizados com hiper-parâmetros e avaliados.
No final da etapa, a ferramenta entrega **(a)** o modelo treinado e serializado no formato ".pkl"; **(b)** o dataset reduzido de características; e **(c)** o relatório de desempenho do modelo. 
- [Grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Optuna](https://github.com/optuna/optuna)

## Instalação 

* Passo 1: clonar o repositório
* Passo 2: instalar os requisitos
```bash
$ ./setup.sh
```

## Relação de algumas dependências
- python3 (versão 3.8.10)
- sklearn==0.0
- scikit-learn==1.1.2 
- pandas==1.3.4
- matplotlib==3.5.0
- optuna==2.10.0
- mlxtend==0.19.0
- jupyter==1.0.0
- termcolor==1.1.0

A relação completa das dependências está disponível em **requirements.txt**.

## Exemplos de possíveis erros e soluções de instalação 
- **Erro** de instalação de dependências  (e.g., *A new release of pip available: 22.1.2 -> 22.2.2*)
	- **Solução**: 
    ```bash
    $ python3 -m pip install --upgrade pip
    ```
- **Erro** de módulo não encontrado (e.g., *Import Error: No module named numpy*)
	- **Solução**:
    ```bash
    $ python3 -m pip install numpy~=1.22.3
    ```

## Parâmetros de entrada (opções de utilização)

```bash
Opções:
  --about               retorna informações do desenvolvedor
  --help                exibe as opções de parâmetros de entrada
  --dataset             dataset (e.g., datasets/DrebinDatasetPermissoes.csv)
  --use-select-features seleção de características (e.g., permissions, api-calls, mult-features )                       
  --sep                 separador utilizado no dataset (valor padrão ",")
  --class-column        nome da coluna que determina a classificação do aplicativo (valor padrão "class")
  --output-results      nome do arquivo de saída das métricas (valor padrão "droidautoml_results.csv")
  --output-model        nome do arquivo de saída para o modelo treinado e serializado em formato ".pkl" (valor padrão "model_serializable.pkl")
```

## Exemplos de utilização

Dataset contendo apenas permissões: 
```bash
python3 droidautoml.py --dataset datasets/drebin_215_permissions.csv --use-select-features permissions
```
Dataset contendo apenas chamadas de API:
```bash
python3 droidautoml.py --dataset datasets/drebin_215_api_calls.csv --use-select-features api-calls
```
Dataset contendo múltiplos tipos de características:
```bash
python3 droidautoml.py --dataset datasets/drebin_all.csv --use-select-features mult-features
```

## Ambiente de testes

A ferramenta foi instalada e testada no seguinte ambiente:
- Notebook Intel(R) Core(TM) i7-1185G7 3.00GHz da geração 11 e memória RAM de 32GB
- Sistema operacional Microsoft Windows 10 64 bit 
- Máquina Virtual [VirtualBox](https://www.virtualbox.org/) (versão 6.1.26 r145957 - Qt5.6.2)
    - 4 vCPUs 
    - 16GB de RAM
    - Linux Ubuntu 20.04.3 LTS 64 bit
    - kernel versão 5.15.0-43-generic
    - GNOME versão 42.0
    - Python versão 3.8.10

