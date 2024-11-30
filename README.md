# TCC---DLTS-para-clique-maxima
Executado em um ambiente anaconda com python 3.9

Execute pip install -r requirements.txt

Arquivo criarGrafos.py cria uma quantidade determinado via parametros de grafos aleatorios e salva na pasta "graphs"

Arquivo run_clisat.sh é um script shell que executa um solucionador sat que retorna a clique maxima para os arquivos gerados com criarGrafos.py e então salva essas soluções na pasta "cliques"

É necessario o arquivo o executavel CliSAT na pasta bin do repositorio https://github.com/psanse/CliSAT
Coloque o executavel na mesma hierarquia que o script sh. O arquivo executavel deve ser renomeado apenas para CliSAT.
Lembrando que o script foi escrito para a versão atual do CliSAT no repositório. Caso haja alterações nos parametros do CliSAT em versões futuras será necessário atualizar o script.

O arquivo criarInstancias então utiliza os grafos e as cliques maximas para criar as instancias de treino que serao usadas para alimentar o modelo de machine learn. As instancias de treino ficam salvas na pasta train_graphs. Para entender as instancias de treino leia o meu TCC em por --link aqui--

O arquivo treinarModelos.ipynb contém o código para gerar os modelos. Esse código foi baseado no trabalho https://github.com/ahottung/DLTS com as necessarias adaptações para o meu problema particular.

Os grafos especificos usados para o treino podem ser encontrados em https://drive.google.com/drive/folders/1IThdxR3MoQTL_URcS41trmpAIAQuLuLv?usp=sharing

Os modelos pré-treinados e usados no testes podem ser encontrados em https://drive.google.com/drive/folders/1uKRZ8LJlOg_74K2MNG8Wt0Q8kAMksyhG?usp=sharing

O arquivo testarModelos.ipynb contem a implementação do algoritmo de branch and bound usando a rede neural, bem como a comparação com os outros algoritmos usandos para a comparação nesse estudo.
