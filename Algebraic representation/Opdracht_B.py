"""
MNIST opdracht B: "Conv Dense"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je alleen nog als laatste layer een dense layer.

De opdracht bestaat uit drie delen: B1 tm B3.

Er is ook een impliciete opdracht die hier niet wordt getoetst
(maar mogelijk wel op het tentamen):
Zorg ervoor dat je de onderstaande code volledig begrijpt.

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
print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.show()
plt.grid(False)

inputShape = x_test[0].shape

# show the shape of the training set and the amount of samples
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1). 
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# This time, we don't flatten the images, because that would destroy the 
# locality benefit of the convolution layers.

# convert class vectors to binary class matrices (one-hot encoding)
# for example 3 becomes (0,0,0,1,0,0,0,0,0,0)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Opdracht B1: 
    
Voeg als hidden layers ALLEEN Convolution en/of Dropout layers toe aan het onderstaande model.
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0,97.

Voorbeelden van layers:
    layers.Dropout(getal)
    layers.Conv2D(getal, kernel_size=(getal, getal), padding="valid" of "same")
    layers.Flatten()
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
je ervaringen daarbij en probeer die ervaringen te verklaren.

Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_B.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            
            # Convolutional layers
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.Dropout(0.25),
            
            # MaxPooling layer
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten the output
            layers.Flatten(),
            
            # Output layer
            layers.Dense(units=num_classes, activation='sigmoid')
        ]
    )
    return model


model = buildMyModel(x_train[0].shape)
model.summary()



"""
Opdracht B2:
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

"""

"""
Opdracht B3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.

Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320

 dropout (Dropout)           (None, 28, 28, 32)        0

 conv2d_1 (Conv2D)           (None, 28, 28, 64)        18496

 dropout_1 (Dropout)         (None, 28, 28, 64)        0

 max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0
 )

 flatten (Flatten)           (None, 12544)             0

 dense (Dense)               (None, 10)                125450

=================================================================
Total params: 144,266
Trainable params: 144,266
Non-trainable params: 0

Formule aantal parameters conv2d: (Aantal filters x kernelgrootte x inputkanaal) + bias
Formule aantal parameters dense: (Aantal input neuronen x aantal output neuronen) + aantal output neuronen

conv2d: 
Input Shape: (None, 28, 28, 1) afkomstig van de invoerlaag
Output Shape: (None, 28, 28, 32)
Berekening: conv2d past 32 filters toe op elke input van 3x3, wat
resulteert in een output feature map van 28x28 met 32 filters.
Het aantal parameters voor deze laag is: 
(32 filters x (3x3) kernelgrootte x 1 inputkanaal) + 32 bias = 320 parameters
(32 x 9 x 1) + 32 = 320 parameters

dropout:
De Dropout-laag voert geen wijzigingen toe aan de vorm van de feature maps.
Het schakelt wel de 25% van de elementen uit, wat resulteert in een uitgedunde representatie.

conv2d_1:
Input Shape: (None, 28, 28, 32) 
Output Shape: (None, 28, 28, 64)
Berekening: conv2d_1 past 64 filters toe op elke input van 3x3, wat
resulteert in een output feature map van 28x28 met 64 filters.
Het aantal parameters voor deze laag is:
(64 filters x (3x3) kernelgrootte x 32 inputkanaal) + 64 bias = 18.496 parameters

dropout_1:
Wederom voert de Dropout-laag geen wijzigingen toe aan de vorm van de feature maps.
Het schakelt wel de 25% van de elementen uit, wat resulteert in een uitgedunde representatie.

max_pooling2d:
Input Shape: (None, 28, 28, 64)
Output Shape: (None, 14, 14, 64)
Berekening: max_pooling2d verdeelt de ruimtelijke dimensies van de feature maps
door de opgegeven poolgrootte (2x2). Hierdoor worden de dimensies gehalveerd (28/2 = 14)
en blijven de 64 kanalen behouden. Dit leidt tot een halvering van het aantal pixels
en helpt bij het verhogen van de ruimtelijke invariantie.

flatten:
Input Shape: (None, 14, 14, 64)
Output Shape: (None, 12544)
Berekening: De flatten laag herstructureert de input feature maps naar een 1D-vector.
In dit geval wordt de dimensie 14x14x64 omgezet naar een vector van lengte 12.544.
14x14x64 = 12.544

dense:
Input Shape: (None, 12544)
Output Shape: (None, 10)
Berekening: De Dense-laag bevat 10 neuronen, wat overeenkomt met het aantal klassen in de dataset(0-9).
Het aantal parameters voor deze laag is 12544 inputkenmerken x 10 neuronen + 10 bias termen = 125.450

Totale aantal parameters = 320 + 18.496 + 125.450 = 144.266
"""

"""
# Train the model
"""
batch_size = 2048 # Larger often means faster training, but requires more system memory.
                  # if you get allocation accord
epochs = 1000    # it's probably more then you like to wait for,
                  # but you can interrupt training anytime with CTRL+C

learningrate = 0.01
#loss_fn = "categorical_crossentropy" # can only be used, and is effictive for an output array of hot-ones (one dimensional array)
loss_fn = 'mean_squared_error'     # can be used for other output shapes as well. seems to work better for categorical as well..

optimizer = keras.optimizers.Adam(lr=learningrate)
model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("\nx_train_flat.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.
learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01
if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_B_weights.h5" here.

print ("\n");
print (ViewTools_NN.getColoredText(255,255,0,"Just type CTRL+C anytime if you feel that you've waited for enough episodes."))
print ("\n");
# # NB: validation split van 0.2 ipv 0.1 gaf ook een boost: van 99.49 naar 99.66
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])    
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

model.summary()
model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print("\nFirst test sample: predicted output and desired output:")
print(prediction[0])
print(y_test[0])

# study the meaning of the filtered outputs by comparing them for
# a few samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer                 # this time, I select the last layer, such that the end-outputs are visualised.
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat = None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
