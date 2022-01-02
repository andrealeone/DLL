
# DLL Project
# University of Trento
# 
# A. Leone, A. E. Piotti
# August-December 2021
#

import os
import pandas            as pd
import numpy             as np
import scipy             as sp
import lpips
import matplotlib.pyplot as plt
import random
import collections
import warnings

import networks
import torch
import sklearn.utils
import sklearn.preprocessing

from sklearn.metrics import confusion_matrix, precision_score
from tqdm.notebook   import tqdm


# TASK 01

def notebook():

    warnings.filterwarnings('ignore')

def load_resources(image_path='../dataset/train/', annotations_path='../dataset/annotations_train.csv'):

    images = sorted([(image_path + file) for file in os.listdir(image_path)])
    annotations = pd.read_csv(annotations_path, index_col='id').sort_values(by='id')

    return images, annotations


# main utilities

def train(model, train_set, criterion, optimizer, device=torch.device('cpu'), epochs=2, li=500, monobatch=False):
    
    if device != torch.device('cpu'):
        print(device)
    
    print(optimizer)
    print(criterion)
    print('\nTRAINING')
    
    model.train()
    
    performance = list()
    
    for epoch in range(epochs):
        
        current_loss = 0.0
        for i, data in tqdm(list( enumerate(train_set) )):
            
            x = data[0].float().to(device)
            y = torch.tensor([data[1]]).to(device)
            
            optimizer.zero_grad()
            
            if monobatch:
                channels, height, width = x.shape
                output = model(x.reshape(1, channels, height, width))
            else:
                output = model(x).to(device)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            performance.append( (epoch, loss.item()) )
            
            if i % li == (li - 1):
                print('loss  %5d:  %.3f' % (i + 1, current_loss / li))
                current_loss = 0.0
    
    return performance

def test(model, test_set, device=torch.device('cpu'), monobatch=False):
    
    print('\nTESTING')
    
    model.eval()
    
    with torch.no_grad():
        
        rl = list()
        
        for data in tqdm( test_set ):
            
            x = data[0].float().to(device)
            y = data[1]
            
            if monobatch:
                channels, height, width = x.shape
                output = model( x.reshape(1, channels, height, width) )
            else:
                output = model(x).to(device)
            prediction = [ float(v) for v in output[0] ]
            result = prediction.index( max(prediction) )
            
            rl.append((y, result, prediction))
    
    t = [ float(y) for y,r,_ in rl ]
    p = [ float(r) for y,r,_ in rl ]
    
    confusion_matrix(t,p)
    
    ls = [ 'accuracy',    'precision',    'recall'     ]
    ms = [  accuracy(t,p), precision(t,p), recall(t,p) ]
    
    for i,m in enumerate(ms) : print( '{:<12}{}'.format( ls[i], m ) )
    print('{:.16f} | {:.7f} | {:.7f}'.format(ms[0], ms[1], ms[2]))
    
    return t, p, ms, rl

def save(model, name, directory='./'):
    torch.save(model.state_dict(), '{}/{}'.format(directory, name))


# multi-classifier utilities

def precision_for_label(label, yt, yp):
    return precision_score(yt, yp, average='binary', zero_division=0, pos_label=label)

def recall_for_label(label, yt, yp):
    return precision_score(yt, yp, average='binary', zero_division=0, pos_label=label)


# metrics

def cosine_similarity_between(a, b):
    return np.dot(a, b) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))

def accuracy(t, p):
    return sklearn.metrics.accuracy_score   (t, p)

def precision(t, p):
    return sklearn.metrics.precision_score  (t, p, average='macro')

def recall(t, p):
    return sklearn.metrics.recall_score     (t, p, average='macro')

def present_metrics(t, p):
    
    ls = [ 'accuracy',    'precision',    'recall' ]
    ms = [  accuracy(t,p), precision(t,p), recall(t,p) ]
    
    for i,m in enumerate(ms) : print( '{:<12}{}'.format( ls[i], m ) )
    
    return ms

def confusion_matrix(t, p):
    
    cm = sklearn.metrics.confusion_matrix(t,p)
    
    _, ax = plt.subplots()
    
    ax.matshow(cm, cmap=plt.cm.Greys, alpha=0.3)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('p')
    plt.ylabel('t')
    
    plt.show()
    
    return cm

def plot_image(image, fs=(16,8)):

    plt.figure( figsize=fs ) 

    frm = plt.gca()
    frm.axes.get_xaxis().set_visible(False)
    frm.axes.get_yaxis().set_visible(False)
    
    plt.imshow( image.int().permute(1,2,0) ) 
    plt.show()

def plot_images(images, nr=5, fs=None): 
    
    if fs is None:
        fs = (10,len(images)-5)
    
    fig, axis = plt.subplots(
        int(len(images)/nr), nr, figsize=fs
    ) 

    for i in range( len(images) ):
        axis[int(i / 5), int(i % 5)].imshow(
            images[i].int().permute(1,2,0)
        ) 

    for axis in fig.axes:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.show()

def plot_model(performance, ln=100):
    
    pf = np.mean( 
        np.array([ p for _,p in performance ])
            .reshape(-1, int( len(performance)/ln )), axis=1
    )
    
    plt.step( range( len( pf ) ), pf )
    plt.show()

def plot_performance(performance_dat, lims='absolute', lm=None):
    
    with open(performance_dat, 'r') as file: 
        performance = np.array([
            float(line)
            for line in file.read().splitlines()
            if line != ''
        ])
    
    if lm is not None:
        performance = performance[-lm:]
    
    plt.figure(figsize=(12, 6)) 
    
    frm = plt.gca()
    frm.axes.get_xaxis().set_visible(False)
    
    plt.step(range(len( performance )), performance, c='crimson') 
    
    if lims == 'absolute':
        plt.ylim(0.0, 1.2)
    if lims == 'relative':
        plt.ylim(min(performance) - 0.01, max(performance) + 0.01)
    if type(lims) == tuple:
        plt.ylim(lims[0], lims[1])
    
    plt.show()

def prediction_accuracy(z, T):
    r = collections.Counter(z == T) 
    return r[True] / (r[True] + r[False])


# other utilities

def split_dataset(data,split_val):
    
    train_set = list()
    test_set  = list()
    
    for _,v in data[:split_val] : train_set.extend(v)
    for _,v in data[split_val:] : test_set .extend(v)
    
    random.Random(3).shuffle(train_set)
    random.Random(3).shuffle(test_set)
    
    return train_set, test_set

def split(dataset, label, reduce=True):

    results   = list()
    positives = list()
    negatives = list()

    for x,y in dataset:

        if y == label:
            positives.append([x[0], 1])

        else:
            negatives.append([x[0], 0])

    random.shuffle( negatives )

    results.extend( positives )

    if reduce:
        results.extend( negatives[:len(positives)] )
    else:
        results.extend( negatives )

    random.shuffle( results )

    return results

def inspect_dataset(data, train_set, test_set):
    
    data_set = list()
    for _,v in data : data_set.extend(v)
    
    print('data:    ', collections.Counter([y for _,y in data_set]))
    print('train:   ', collections.Counter([y for _,y in train_set]))
    print('test:    ', collections.Counter([y for _,y in test_set]))

def describe_sets(train_set, test_set):
    
    print('train:   ', collections.Counter([y for _,y in train_set]))
    print('test:    ', collections.Counter([y for _,y in test_set]))

def describe_triplet(triplet):

    plt.figure( figsize=(16, 8) ) 
    plt.imshow( triplet[0][0].int().permute(1,2,0) ) 
    plt.show()
    
    for s,t in zip(['Anchor', 'Positive', 'Negative'], triplet): 
        print( ('{}\n{}\n').format(s, t.int().detach().numpy()) )

def split_in_shapes(tensor, device=torch.device('cpu')): 
    
    results = []
    shapes  = [
        (0,128), (0,35), (20,80), (40,100), (60,120)
    ]
    
    for a,b in shapes:
        
        shape = [] 
        for c in tensor[0]:
            shape.append( c[a:b].numpy() ) 
        
        results.append( torch.tensor([shape]).float().to(device) ) 
    
    return results

def compare_image_attributes(image, labels, r1, r2):

    plt.figure( figsize=(16, 8) ) 
    plt.imshow( image[0].int().permute(1,2,0) ) 
    plt.show()

    print( np.array([ 
        [ '{:<12}:'.format(annotation), r1[i], r2[i] ]
        for i, annotation
        in enumerate(labels)
    ]))


# assembler

def load_model_with_weights(network, model_name, input_size, output_size):
    
    attribute, model_type, model_version, _ = model_name.split('.')
    
    model = network(input_size, output_size)
    model.load_state_dict( torch.load('./models/' + model_name) )
    model.eval()
    
    print('{0:<20} {1:<22} v{2}'.format(attribute, model_type, model_version))
    return model


# super-model

def super_model(_input, src): 
    
    vector      = list()
    tensor_mode = split_in_shapes(_input)

    def add_simple_model_result_to(vector, attribute, shape): 
    
        vector.extend([
            np.argmax(

                src[attribute](shape)
                    .detach().numpy()

            )
        ])
    
    def add_ensemble_model_result_to(vector, attribute, shape, pad=False): 
        
        ms = src[attribute]['models']
        i  = np.argmax(

            src[attribute]['ensemble'](
                torch.tensor([ 

                    model( shape ).detach().numpy()[0]
                    for model in ms

                ]).reshape(-1)
            ).detach().numpy()

        )
        
        if pad:
            array = np.zeros( len(ms) - 1, dtype=int )
            
            if i > 0:
                array[i - 1] = 1
            
            vector.extend(array)
        
        else:
            vector.extend([ i ])
    
    add_ensemble_model_result_to(vector,
        attribute='age', shape=tensor_mode[0]
    )
    
    attribute_refs = [
        ('backpack', 2), ('bag', 2), ('handbag', 3),
        ('clothes', 4), ('down', 4), ('up', 2),
        ('hair', 1), ('hat', 1), ('gender', 0)
    ]
    
    for attribute, s in attribute_refs:
        add_simple_model_result_to(vector,
            attribute, shape=tensor_mode[s]
        )
    
    add_ensemble_model_result_to(vector,
        attribute='colors_up', shape=tensor_mode[2], pad=True
    )
    
    add_ensemble_model_result_to(vector,
        attribute='colors_down', shape=tensor_mode[4], pad=True
    )
    
    return torch.tensor(vector).float()


# TASK 02

def reid_model(q_id, images, attributes, arrays, siamese_net, q_res_lim=30, q_res_trs=3):

    test_ids_iloc = list( attributes['test'].keys() ) 

    q          = attributes['queries'][q_id] - 1
    q_img      = images['queries'][q_id][0].int() 
    q_res      = sorted([ 
        [ test_ids_iloc[_i], _dist[0] ] for _i, _dist in enumerate(
            sklearn.metrics.pairwise.cosine_similarity(arrays, [q])
        )
    ], key=lambda x : x[1], reverse=True)

    q_res2     = collections.Counter()
    q_embed    = siamese_net( images['queries'][q_id] )[0].detach().numpy() 
    q_embeds   = [ 
        siamese_net( images['test'][_id] )[0].detach().numpy()
        for _id,_ in q_res[:q_res_lim]
    ]

    for _ in range(q_res_trs):
        q_res2.update( dict([ 
            [ _id, int((q_res_lim-_i)/3) ] for _i,(_id,_) in enumerate(

                [ q_res[_i][0],_dist  ] for _i,_dist in sorted([ 
                    [ _i,float(_dist) ] for _i,_dist in enumerate(
                        sklearn.metrics.pairwise.cosine_similarity( q_embeds, [ q_embed ] )
                    )
                ], key=lambda x:x[1], reverse=True)

            )
        ]))

    q_res2     = q_res2.most_common()
    q_res3     = dict() 
    lpips_mods = dict()
    lpips_nets = [ 'alex', 'vgg' ]

    for net in lpips_nets:
        
        q_res3[ net ]     = list()
        lpips_mods[ net ] = lpips.LPIPS(
            net=net, eval_mode=True,
            pretrained=True, verbose=False
        )

    for _id,_ in q_res2[:q_res_lim]:
        for net, mod in lpips_mods.items():
        
            img  = images['test'][_id][0].int()
            loss = float( mod.forward( q_img, img ) )

            q_res3[ net ].append([ _id, loss ])

    q_res3C = collections.Counter()
    for net in lpips_nets: 
        
        q_res3C.update(
            collections.Counter(dict([ 
                [ _id,_i ] for _i,(_id,_) in enumerate( sorted(
                    q_res3[net], key=lambda x:x[1], reverse=True
                ))
            ]))
        )

    q_res3 = q_res3C.most_common()

    return q_res3


# legacy links

def CNN(size):
    return networks.CNN(size)

