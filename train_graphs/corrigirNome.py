import os

# Diretório onde os arquivos estão localizados
diretorio = './'

# Loop através dos arquivos no diretório
for filename in os.listdir(diretorio):
    # Verifica se o arquivo segue o padrão esperado
    if filename.startswith('instance_graph_') and filename.endswith('.dimacs'):
        # Divide o nome do arquivo para extrair as partes necessárias
        partes = filename.split('_')
        numero = partes[2]  # Extrai o número
        novo_nome = f'graph_{numero}_var0.dimacs'  # Cria o novo nome

        # Caminho completo para os arquivos antigo e novo
        caminho_antigo = os.path.join(diretorio, filename)
        caminho_novo = os.path.join(diretorio, novo_nome)

        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f'{filename} renomeado para {novo_nome}')
