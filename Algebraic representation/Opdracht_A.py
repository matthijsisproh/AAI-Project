"""
MNIST opdracht A: "Only Dense"      (by Marius Versteegen, 2021)
Bij deze opdracht ga je handgeschreven cijfers klassificeren met 
een "dense" network.

De opdracht bestaat uit drie delen: A1, A2 en A3 (zie verderop)

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
    
--> Zorg ervoor dat je de onderstaande code volledig begrijpt. <--
******************************************************************

Tip: stap in de Debugger door de code, en bestudeer de tussenresultaten.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MavTools_NN import ViewTools_NN

#tf.random.set_seed(0) #for reproducability

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#print(x_test[0])
print("Showing first test-image\n")
# plt.figure()
#plt.imshow(x_test[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples\n")

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# we can flatten the images, because we use Dense layers only
x_train_flat = x_train.reshape(x_train.shape[0], inputShape[0]*inputShape[1])
x_test_flat  = x_test.reshape(x_test.shape[0], inputShape[0]*inputShape[1])

"""
Opdracht A1: 
    
Voeg ALLEEN Dense en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,95.

Voorbeelden van layers:
    layers.Dropout( getal ),
    layers.Dense(units=getal, activation='sigmoid'),
    layers.Dense(units=getal, activation='relu'),

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

[ervaring: Zonder GPU, op I5 processor < 5 min, 250 epochs bij batchsize 4096]

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_A.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.

        [
            keras.Input(shape=inputShape),
            layers.Dropout(.1),
            layers.Dense(units=100, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=20, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Dropout(.1),
            layers.Dense(units=200, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=100, activation='sigmoid'),
            layers.Dropout(.1),
            layers.Dense(units=10, activation='sigmoid'),
        ]
    )
    return model

model = buildMyModel(x_train_flat[0].shape)
model.summary()
print("Ik ben hier")

"""
Opdracht A2: 

Verklaar met kleine berekeningen het aantal parameters dat bij elke laag
staat genoemd in bovenstaande model summary.

Formule dense laag = (aantal input neurons * aantal output neurons) + aantal output neurons

MNIST Shape = 28 x 28 x 1, dus aantal input neurons van dense = 28 x 28 x 1 = 784
Aantal output neurons dense = 200

dense = (784 x 200) + 200 = 157.000

Aantal input neurons van dense_1 = aantal output neurons van dense = 200
Aantal output neurons van dense_1 = 200
dense_1 = (200 x 100) + 100 = 20.100

Aantal input neurons van dense_2 = aantal output neurons van dense_1 = 100
dense_2 = (100 x 10) + 10 = 1.010
"""

"""
# Train the model
"""
batch_size = 10   # Larger can mean faster training (especially when using the gpu: 4096), 
                  # but requires more system memory. Select it properly for your system.
                    
epochs = 1000     # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'        # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model, or if you want to restart using different (random) weights.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_A_weights.h5" here.
print (ViewTools_NN.getColoredText(255,255,0,"\nJust type CTRL+C anytime if you feel that you've waited for enough episodes.\n"))

try:
    model.fit(x_train_flat, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard\n")

"""
# Evaluate the trained model
"""
score = model.evaluate(x_test_flat, y_test, verbose=0)
print("Test loss:", score[0])       # 0.0394
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]), "\n")   # 0.9903

model.summary()
#model.save_weights('myWeights.h5')

prediction = model.predict(x_test_flat)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0],"\n")

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer = 1               # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

"""
Opdracht A3: 

1 Leg uit wat de betekenis is van de output images die bovenstaand genenereerd worden met "printFeatureMapsForLayer".

De outputbeelden gegenereerd door de functie `printFeatureMapsForLayer` 
laten zien hoe de activaties van de neuronen in de gekozen laag reageren 
op de invoerbeelden. Elk beeld in de gegenereerde reeks toont de activaties 
van een specifiek neuron in de laag voor verschillende invoerbeelden. Dit geeft ons inzicht 
in de representatie van de invoergegevens op dat specifieke niveau.

2 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de eerste dense layer meegeeft.

Als we de index van de eerste Dense-laag aan de functie "printFeatureMapsForLayer" 
zouden geven, zouden we geen betekenisvolle outputbeelden krijgen. De eerste Dense-laag 
voert een lineaire transformatie uit op de invoer en heeft geen ingebouwde niet-lineariteit. 
Hierdoor zouden de activaties van de neuronen in de laag niet veel verschillen tonen 
en zouden de outputbeelden mogelijk niet informatief zijn.

Door hier naar te kijken, kunnen we een gevoel krijgen over welke representaties en kenmerken 
het model heeft geleerd te herkennen. Dit gevoel kunnen we gebruiken voor het verbeteren van het model.


3 Leg uit wat de betekenis zou zijn van wat je ziet als je aan die functie de index van de tweede dense layer meegeeft.

Als we de index van de tweede Dense-laag aan de functie "printFeatureMapsForLayer" zouden geven, 
zouden we de activaties van de neuronen in die laag kunnen visualiseren. 

Aangezien de tweede Dense-laag non-lineariteit introduceert door de sigmoid-activatiefunctie, zouden 
de outputbeelden ons laten zien hoe de neuronen in die laag verschillende patronen in de invoergegevens detecteren. 

Net als bij de vorige vraag kunnen we hier ook een gevoel krijgen van hoe goed het model classificeert.

"""
