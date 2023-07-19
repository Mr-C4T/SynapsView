
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


def synapsView(model,verbose=False,label=True,size=1):


    colorListe=["c","y"]
    valueListe=[-1,1]    
    
    # Graph param
    plt.rcParams['figure.facecolor'] = '#111111'
    plt.rcParams['axes.facecolor'] = '#111111'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams["figure.figsize"] = (28*size, 18*size)
    plt.axis('off')

    # Récupérer les poids du modèle
    Wgraph = copy.deepcopy(model.coefs_)
    # Ajouter la couche de sortie à la fin de la liste weights
    out=model.n_outputs_
    Wgraph.append(np.zeros((out, Wgraph[-1].shape[1])))  
    
    # Trouver la plus grande taille de couche
    max_layer_size = np.max(model.hidden_layer_sizes)
    if(len(Wgraph[0])>max_layer_size):
       max_layer_size=len(Wgraph[0])

    if label:
        for i in range(0,len(colorListe)):
            plt.scatter(i*0.5,-1, c=colorListe[i],s=100*size,label=valueListe[i])
            plt.text(i*0.5+0.1,-1,valueListe[i] ,fontsize=33*size)

    cmap = LinearSegmentedColormap.from_list(valueListe, colorListe)

    # Pour chaque couche cachée...
    for X, layer in tqdm(enumerate(Wgraph)):
        #Pour chaque neurones
        #for Y,neuron in enumerate (layer):
        for Y,neuron in enumerate (layer):
            #Dessiner le neurone
            plt.scatter(X, Y+(max_layer_size/2-len(Wgraph[X])/2), c='white',s=50*size,zorder=10)
            #Pour chaque poids
            for N,W in enumerate(neuron):
                if X == len(Wgraph)-1:
                    break
                if verbose:
                    print("Weight ",N+1," of Neuron ",Y+1," in layer ",X+1," =",W)
                    #print("Weight ",N+1," of Neuron ",Y+1," in layer ",X+1," =",Wgraph[X][Y][N])
                # Dessiner le poids
                plt.plot([ X, X+1], [Y+(max_layer_size/2-len(Wgraph[X])/2), N+(max_layer_size/2-len(Wgraph[X+1])/2)], c=cmap(W),linewidth='1')
                
    # Affichage de l'image
    plt.savefig('./nn.png')  
    plt.show() 


X=[[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
y=[1,0,1,0]

# Split the data into training and test sets, with a test size of 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Créer un classifieur MLP
mlp = MLPClassifier(hidden_layer_sizes=(12,14,10), max_iter=50 )

# Entraîner le classifieur MLP sur les données d'entraînement
mlp.fit(X_train, y_train)

synapsView(mlp)

print("\nScore:",mlp.score(X_train, y_train))

# Make predictions on the test set
predictions = mlp.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, predictions)

print(cm)