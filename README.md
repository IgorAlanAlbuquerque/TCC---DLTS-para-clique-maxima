# TCC---DLTS-para-clique-maxima
Arquivo criarGrafos.py cria uma quantidade determinado via parametros de grafos aleatorios e salva na pasta "graphs"

Arquivo run_clisat.sh é um script shell que executa um solucionador sat que retorna a clique maxima para os arquivos gerados com criarGrafos.py e então salva essas soluções na pasta "cliques"

É necessario o arquivo o executavel CliSAT na pasta bin do repositorio https://github.com/psanse/CliSAT
Coloque o executavel na mesma hierarquia que o script sh

O arquivo criarInstancias então utiliza os grafos e as cliques maximas para criar as instancias de treino que serao usadas para alimentar o modelo de machine learn. As instancias de treino ficam salvas na pasta train_graphs. Para entender as instancias de treino leia o meu TCC que esta nesse repositorio.

O arquivo treinarModelos.py contém o código para gerar os modelos. Esse código foi baseado no trabalho https://github.com/ahottung/DLTS com as necessarias adaptações para o meu problema particular.
