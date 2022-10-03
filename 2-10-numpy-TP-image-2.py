# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype,
#       -toc
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   nbhosting:
#     title: suite du TP simple avec des images
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p22/python-numerique/main/notebooks/_static/style.html")



# %% [markdown]
# # suite du TP simple avec des images
#
# merci à Wikipedia et à stackoverflow
#
# **le but de ce TP n'est pas d'apprendre le traitement d'image  
# on se sert d'images pour égayer des exercices avec `numpy`  
# (et parce que quand on se trompe ça se voit)**

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown] {"tags": ["framed_cell"]}
# **notions intervenant dans ce TP**
#
# sur les tableaux `numpy.ndarray`
#
# * `reshape()`, tests, masques booléens, *ufunc*, agrégation, opérations linéaires sur les `numpy.ndarray`
# * les autres notions utilisées sont rappelées (très succinctement)
#
# pour la lecture, l'écriture et l'affichage d'images
#
# * utilisez `plt.imread`, `plt.imshow`
# * utilisez `plt.show()` entre deux `plt.imshow()` dans la même cellule
#
# **note**
#
# * nous utilisons les fonctions de base sur les images de `pyplot` par souci de simplicité
# * nous ne signifions pas là du tout que ce sont les meilleures  
# par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
# alors que la fonction `save` de `PIL` le permet
#
# * vous êtes libres d'utiliser une autre librairie comme `opencv`  
#   si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte
#
# **n'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# ## Création d'un patchwork

# %% [markdown]
# 1. Le fichier `rgb-codes.txt` contient une table de couleurs:
# ```
# AliceBlue 240 248 255
# AntiqueWhite 250 235 215
# Aqua 0 255 255
# .../...
# YellowGreen 154 205 50
# ```
# Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
# Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.
# <br>
#
# 1. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
# `'Red'`, `'Lime'`, `'Blue'`
# <br>
#
# 1. Faites une fonction `patchwork` qui  
#
#    * prend une liste de couleurs et la structure donnant le code des couleurs RGB
#    * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
#    * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
#    si besoin de compléter l'image mettez du blanc  
#    (`numpy.indices` peut être utilisé)
# <br>
# <br>   
# 1. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.
# <br>
#
# 1. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
# même chose pour des jaunes  
# <br>
#
# 1. Appliquez la fonction à toutes les couleurs du fichier  
# et sauver ce patchwork dans le fichier `patchwork.jpg` avec `plt.imsave`
# <br>
#
# 1. Relisez et affichez votre fichier  
#    attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels
#    
# vous devriez obtenir quelque chose comme ceci
# <img src="patchwork-all.jpg" width="200px">

# %%

with open ('rgb-codes.txt') as code:
    dico={ l.split()[0]: np.array(l.split()[1:], dtype=np.uint8) for l in code.readlines()}
dico


dico['Red']
dico['Lime']
dico['Blue']

def patchwork(l):
    a=len(l)
    n=int(np.sqrt(a))
    if not n**2==a:
        n+=1
    im=np.ones(shape=(10*n,10*n,3), dtype=np.uint8)*255
    for k in range (a):             # et oui une boucle for ...
        ligne=k//n
        colonne=k-n*ligne
        im[10*ligne:10*ligne+10,10*colonne:10*colonne+10,::]=l[k]
    return im



l=np.random.randint(256, size=(100,3),  dtype=np.uint8)
im=patchwork(l)
plt.imshow(im);
plt.show()



def distance(l1,l2):
    s=0
    for i in range(3):
        s+= abs(l1[i]-l2[i])
    return s 
        
        
l=[ dico[i] for i in dico if distance(dico[i],[255,255,255])<=200]
plt.imshow( patchwork(l));
plt.show()


l=[ dico[i] for i in dico if distance( dico[i],[255,255,0])<=200]
plt.imshow( patchwork(l));
plt.show()


l=[ dico[i] for i in dico ]
im=patchwork(l)
plt.imshow(im);
plt.show()
plt.imsave('patchwork.jpg', im)
im2=plt.imread('patchwork.jpg')
plt.imshow(im2);




# %% [markdown]
# ## Somme des valeurs RGB d'une image

# %% [markdown]
# 0. Lisez l'image `les-mines.jpg`
#
# 1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image  
#
# 2. Affichez l'image (pas terrible), son maximum et son type
#
# 3. Créez un nouveau tableau `numpy.ndarray` en sommant **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image
#
# 4. Affichez l'image, son maximum et son type
#
# 5. Pourquoi cette différence ? Utilisez le help `np.sum?`
#
# 6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
# (de la manière que vous préférez)
#
# 7. Remplacez dans l'image en niveaux de gris,   
# les valeurs >= à 127 par 255 et celles inférieures par 0  
# Affichez l'image avec une carte des couleurs des niveaux de gris  
# vous pouvez utilisez la fonction `numpy.where`
#
# 8. avec la fonction `numpy.unique`  
# regardez les valeurs différentes que vous avez dans votre image en noir et blanc

# %%

image=plt.imread('les-mines.jpg')

im=image[::,::,0]+image[::,::,1]+image[::,::,2]
plt.imshow(im, cmap='Greys');
plt.show()

im=image.sum( axis=2 )

plt.imshow(im, cmap='Greys');
plt.show()
np.max(im) 
type(im)


im2=np.array(im//3, dtype=np.uint8)

im3=np.where(im2>=127,255,0)
plt.imshow(im3, cmap='Greys');
plt.show()

np.unique(im3)

# %% [markdown]
# ## Image en sépia

# %% [markdown]
# Pour passer en sépia les valeurs R, G et B d'un pixel  
# (encodées ici sur un entier non-signé 8 bits)  
#
# 1. on transforme les valeurs $R$, $G$ et $B$ par la transformation  
# $0.393\, R + 0.769\, G + 0.189\, B$  
# $0.349\, R + 0.686\, G + 0.168\, B$  
# $0.272\, R + 0.534\, G + 0.131\, B$  
# (attention les calculs doivent se faire en flottants pas en uint8  
# pour ne pas avoir, par exemple, 256 devenant 0)  
# 1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
# 1. naturellement l'image doit être ensuite remise dans un format correct  
# (uint8 ou float entre 0 et 1)

# %% [markdown]
# **Exercice**
#
# 1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
# la fonction `numpy.dot` doit être utilisée (si besoin, voir l'exemple ci-dessous) 
#
# 1. Passez votre patchwork de couleurs en sépia  
# Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso
# 2. Passez l'image `les-mines.jpg` en sépia   

# %%

mat_transition=np.transpose(np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]))

def sepia(im):
    im2=np.array(im, dtype=float)
    im2=np.dot( im2, mat_transition)
    im2=np.array(np.where(im2>255,255,im2), dtype=np.uint8)
    return im2


patchwork = plt.imread('patchwork.jpg')
plt.imshow( sepia(patchwork))
plt.show()


image=plt.imread('les-mines.jpg')
plt.imshow(sepia(image));


# %% {"scrolled": true}
# INDICE:

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

A.shape, B.shape, C.shape

# %% [markdown]
# ## Exemple de qualité de compression

# %% [markdown]
# 1. Importez la librairie `Image`de `PIL` (pillow)   
# (vous devez peut être installer PIL dans votre environnement)
# 1. Quelle est la taille du fichier 'les-mines.jpg' sur disque ?
# 1. Lisez le fichier 'les-mines.jpg' avec `Image.open` et avec `plt.imread`  
#
# 3. Vérifiez que les valeurs contenues dans les deux objets sont proches
#
# 4. Sauvez (toujours avec de nouveaux noms de fichiers)  
# l'image lue par `imread` avec `plt.imsave`  
# l'image lue par `Image.open` avec `save` et une `quality=100`  
# (`save` s'applique à l'objet créé par `Image.open`)
#
# 5. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
# Que constatez-vous ?
#
# 6. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence  

# %%

from PIL import Image 


image=plt.imread ( 'les-mines.jpg')
print(image.shape)
print(image.size)

image1=plt.imread('les-mines.jpg')
image2=Image.open( 'les-mines.jpg')

print (np.all(np.isclose(image1, image2))) #True

plt.imsave( 'imsave.jpg', image1)
image2.save ('save.jpg', quality=100)

print(plt.imread('imsave.jpg').size)
print(plt.imread('save.jpg').size)

plt.imshow(255-np.abs(np.add(plt.imread('imsave.jpg'),(plt.imread('save.jpg')*(-1))))); 
