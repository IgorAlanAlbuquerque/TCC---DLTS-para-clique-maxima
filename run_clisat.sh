#!/bin/bash

# Diretório contendo os arquivos de entrada
input_dir="graphs"
# Diretório onde os arquivos de saída serão armazenados
output_dir="cliques"
# Caminho para o binário CliSAT
clisat_bin="./CliSAT"
# Limite de tempo em segundos
time_limit=100
# Ordenação dos vértices (1 para DEG-SORT, 2 para COLOR-SORT)
ordering=1

# Criar o diretório de saída se não existir
mkdir -p $output_dir

# Iterar sobre todos os arquivos no diretório de entrada
for filepath in $input_dir/*.dimacs; do
    filename=$(basename -- "$filepath")
    output_filename="clique_${filename}"
    output_path="${output_dir}/${output_filename}"
    
    echo "Processando arquivo: $filepath"

    # Executar o CliSAT
    $clisat_bin "$filepath" $time_limit $ordering 1 > "$output_path"
    
    if [ $? -eq 0 ]; then
        echo "Arquivo processado com sucesso: $filepath"
        if [ -s "$output_path" ]; then
            echo "Arquivo de saída criado: $output_path"
        else
            echo "Falha ao criar o arquivo de saída: $output_path"
        fi
    else
        echo "Erro ao processar $filepath"
    fi
done

echo "Processamento concluído."
