import os
import os.path
import numpy as np
import random
import glob
from keras.models import Sequential
from keras.layers import Dense, Activation, Input,Concatenate
from keras.optimizers import SGD
import datetime
from keras.models import load_model, Model
import logging
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras import optimizers
import h5py
import csv
import time

def parse_file_pointer(fp, tam):
    lines = [ll.strip() for ll in fp]
    ii = 0
    labels = []
    res = []
    numLinhas = 0
    while ii < len(lines):
        line = lines[ii]
        #contando o numero de vertices do grafo
        if "cliqueatual" not in line:
            ii += 1
            numLinhas += 1
            continue

        #pegando a clique atual
        line = line[3:]
        spritado = line.split()
        clique = [int(elem) for elem in spritado[1:]]
        if(numLinhas < tam):
            dif = tam - numLinhas
            clique.extend([0]*dif)

        #criando o vetor de movimento
        line = lines[ii+1]
        sp = line.split()
        mv = int(sp[-1])
        label = [0] * tam
        label[mv] = 1
        labels.append(label)

        #lendo o grafo
        cells = []
        for tt in range(numLinhas, 0, -1):
            cell_line = lines[ii - tt][3:]
            cells.extend([int(float(cc)) for cc in cell_line.split(", ")])
            if(numLinhas < tam):
                dif = tam - numLinhas
                cells.extend([0]*dif)

        #cells = np.reshape(cells,((tam,  -1)))
        #cells = np.transpose(cells)
        #cells = np.reshape(cells, -1)
        res.append(cells)
        ii += (numLinhas+2)
    labels_v = list(range(len(labels),0, -1))
    return (res, labels, labels_v)

def parse_dir(ddir, tam):
    res = []
    labels = []
    labels_v = []
    random.seed(42)
    files = sorted([os.path.basename(ii) for ii in glob.glob("{0}/*.txt".format(ddir))])
    random.shuffle(files)
    random.seed()
    for ff in files:
        with open(os.path.join(ddir,ff), 'r') as fp:
            rr,ll,ll_v = parse_file_pointer(fp, tam)
            print("\nfim instancia\n")
            res.extend(rr)
            labels.extend(ll)
            labels_v.extend(ll_v)
    return res, labels, labels_v


class printbatch(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        logging.info("Epoch: "+ str(epoch))
    def on_epoch_end(self, epoch, logs={}):
        logging.info(logs)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        self.level(sys.stderr)


def learn(data, labels, tam, output_path, shared_layer_multipliers, layer_multipliers, batch_size, learning_rate):
    
    shared_layer_multipliers = [x for x in shared_layer_multipliers if x != 0]

    #camada de entrada
    graph_input = Input(shape=(tam,))
    clique_input = Input(shape=(tam,))
    inputs = [graph_input, clique_input]

    #camadas densas compartilhadas
    for i in range(len(shared_layer_multipliers)):
        shared_dense = Dense(tam*shared_layer_multipliers[i],activation='relu')
        inputs = [shared_dense(inp) for inp in inputs]
    merged_vector = Concatenate(inputs, axis=-1)
    
    #camadas internas
    layer = merged_vector
    for i in range(len(layer_multipliers)):
        layer = Dense(tam*layer_multipliers[i],activation='relu')(layer)
        
    #camada de saida
    output_layer = Dense(tam, activation='softmax')(layer)
    
    #compilar modelo
    model = Model(inputs=[graph_input, clique_input], outputs=output_layer)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #treinar modelo
    now = datetime.now()
    model.fit(np.hsplit(data,tam), labels,nb_epoch= 1000,batch_size=batch_size,validation_split=0.2,verbose=2,
              callbacks=[printbatch(), EarlyStopping(monitor='val_loss', patience=50, verbose=0), ModelCheckpoint(os.path.join(output_path, "models",
                            "dnn_model_" + str(tam) + "_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) + "_"
                            + "_{epoch:02d}-{val_loss:.2f}" + ".h5"),
                            monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])
    model.save(os.path.join(output_path, "dnn_model_" + str(tam)+"_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) +"_"+ ".h5"))


def learn_value(data, labels, tam, output_path, shared_layer_multipliers, layer_multipliers, batch_size, learning_rate):
   
    shared_layer_multipliers = [x for x in shared_layer_multipliers if x != 0]

    #camada de entrada
    graph_input = Input(shape=(tam,))
    clique_input = Input(shape=(tam,))
    inputs = [graph_input, clique_input]

    #camadas densas compartilhadas
    for i in range(len(shared_layer_multipliers)):
        shared_dense = Dense(tam*shared_layer_multipliers[i],activation='relu')
        inputs = [shared_dense(inp) for inp in inputs]
    merged_vector = Concatenate(inputs, axis=-1)
    
    #camadas internas
    layer = merged_vector
    for i in range(len(layer_multipliers)):
        layer = Dense(tam*layer_multipliers[i],activation='relu')(layer)

    #camada de saida
    output_layer = Dense(1)(layer)

    #compilar modelo
    model = Model(inputs=[graph_input, clique_input], outputs=output_layer)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    #treinar modelo
    now = datetime.datetime.now()
    model.fit(np.hsplit(data, tam), labels, nb_epoch=1000, batch_size=batch_size, validation_split=0.2, verbose=2,
              callbacks=[printbatch(), EarlyStopping(monitor='val_loss', patience=50, verbose=0), ModelCheckpoint(os.path.join(output_path, "models",
                            "dnn_value_model_" + str(tam) + "x" +"_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) + "_" +
                            "_{epoch:02d}-{val_loss:.2f}" + ".h5"), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                            mode='auto')])
    model.save(os.path.join(output_path, "dnn_value_model_"+str(tam)+"_"+str(now.day)+"."+str(now.month)+"."+str(now.year)+"_"+".h5"))


def main():
    output_path = "modelos" #caminha onde é salvo o modelo
    labeled_data_dir = "train_graphs" #caminho dos dados para treinar o modelo
    param_v_a_1 = [4, 3, 2] #camadas compartilhadas rede bound
    param_v_a_2 = [3, 2, 2] #camadas ocultas
    param_p_a_1 = [6, 4, 3] #camadas compartilhadas rede brach
    param_p_a_2 = [9, 6, 2] #camadas ocultas
    param_p_b = 512 #batch size
    param_v_b =  512
    param_p_l = 0.001 #taxa de aprendizado
    param_v_l = 0.001 
    tam = 250
    use_value_model = True #se vai treinar a rede de bound

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #treinar rede branch
    res, labels, labels_v = parse_dir(labeled_data_dir, tam)
    print("leu")
    learn(np.array(res), np.array(labels), tam, output_path, param_p_a_1, param_p_a_2, param_p_b, param_p_l)
    print("rede 1 treinada")
    #treinar rede bound
    if use_value_model:
        res, labels, labels_v = parse_dir(labeled_data_dir, tam)
        learn_value(np.array(res), np.array(labels_v), tam, output_path, param_v_a_1, param_v_a_2, param_v_b, param_v_l)
        print("rede 2 treinada")

main()
