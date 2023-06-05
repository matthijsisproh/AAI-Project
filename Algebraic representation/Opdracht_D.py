"""
MNIST opdracht D: "Localisatie"      (by Marius Versteegen, 2021)
Bij deze opdracht passen we het in opdracht C gevonden model toe om
de posities van handgeschreven cijfers in een plaatje te localiseren.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import random
from MavTools_NN import ViewTools_NN
 
# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print(x_test[0])

print("show image\n")
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Now, for a test on the end, let's create a few large images and see if we can detect the digits
# (we do that here because the dimensions of the images are not extended yet)
largeTestImage = np.zeros(shape=(1024,1024),dtype=np.float32)
# let's fill it with a random selection of test-samples:
nofDigits = 40
for i in range(nofDigits):
    # insert an image
    digitWidth  = x_test[0].shape[0]
    digitHeight = x_test[0].shape[1]
    xoffset = random.randint(0, largeTestImage.shape[0]-digitWidth)
    yoffset = random.randint(0, largeTestImage.shape[1]-digitWidth)
    largeTestImage[xoffset:xoffset+digitWidth,yoffset:yoffset+digitHeight] = x_test[i][0:,0:]
    
largeTestImages = np.array([largeTestImage])  # for now, I only test with one large test image.

print("show largeTestImage\n")
plt.figure()
plt.imshow(largeTestImages[0])
plt.colorbar()
plt.grid(False)
plt.show()
matplotlib.image.imsave('largeTestImage.png',largeTestImages[0])


# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
largeTestImages = np.expand_dims(largeTestImages, -1)  # one channel per color component in lowest level. In this case, thats one.

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train,-2),-2)
y_test = np.expand_dims(np.expand_dims(y_test,-2),-2)

"""
Opdracht D1: 

Kopieer in het onderstaande model het model dat resulteerde
in opdracht C. Gebruik evt de weights-file uit opdracht C, om trainingstijd in te korten.
"
Bestudeer de output files: largeTestImage.png en de Featuremaps pngs.
* Leg uit hoe elk van die plaatjes tot stand komt en verklaar aan de hand van een paar
  voorbeelden wat er te zien is in die plaatjes.

  1. largeTestImage.png:
  Dit is de uitvoer van het model voor een testafbeelding. Het toont de classificatie of
  voorspelling van het model voor die afbeelding.
  2. Featuremap pngs:
  Deze afbeeldingen worden gegenereerd door de featuremap lagen van het model. Ze tonen
  de geactiveerde kenmerken op basis van geleerde filters, zoals lijnen, randen, texturen, objecten, etc.
  
  * Als het goed is zie je dat de pooling layers in grote lijnen bepalen hoe groot de 
  uiteindelijke output bitmaps zijn. Hoe werkt dat?
  
  Klopt, deze lagen verminderen de dimensie van de featuremaps door informatie samen te vatten en te
  downsamplen. Dit verkleint de resolutie en behoudt de belangrijkste kenmerken.

  Het gebruik van pooling-lagen verhoogt de performance. Ten opzichte van kleine translaties,
  verbetert robuustheid en generaliseert het model.
  
"""

"""
Opdracht D2:
    
* Met welke welke stappen zou je uit de featuremap plaatjes lijsten kunnen 
genereren met de posities van de cijfers?

Om lijsten met posities van cijfers uit de featuremap plaatjes te genereren, kan een drempelwaarde
worden bepaalt voor de activaties in de featuremap plaatjes. En de posities van pixel waarvan de
activatiewaarde hoger is dan de drempelwaarde identificeren. Deze posities kunnen we vervolgens
in een lijst zetten.

Zoals onderstaand geschreven code:
# Drempelwaarde en detectie van posities
threshold = 0.5
digit_positions = []

for digit in range(10):
    digit_position = []

    # Loop door elke featuremap
    for featuremap_index, featuremap in enumerate(prediction2):
        # Loop door elke pixel in de featuremap
        for row in range(featuremap.shape[0]):
            for col in range(featuremap.shape[1]):
                # Controleer of de intensiteitswaarde hoger is dan de drempelwaarde 
                if featuremap[row, col, digit] > threshold:
                    # Voeg de positie toe aan de lisjt
                    digit_position.append((featuremap_index, row, col))
    
    digit_positions.append(digit_position)

"""

"""
Opdracht D3: 

* Hoeveel elementaire operaties zijn er in totaal nodig voor de convoluties voor 
  onze largeTestImage? (Tip: kijk terug hoe je dat bij opdracht C berekende voor 
  de kleinere input image)
    Voor elke convolutielaag kunnen we het aantal FLOPs berekenen met behulp van de formule:
    FLOPs = (Ho * Wo * Hi * Wi * K * K * Ci * Co) * B
    Ho en Wo: de hoogte en breedte van de uitvoerfeaturemap
    Hi en Wi: de hoogte en breedte van de invoerfeaturemap
    K: de kernelgrootte
    Ci en Co: het aantal kanalen (filters) in de invoer- en uitvoerfeaturemap
    B: het aantal beelden (batchgrootte)
    
    Dus voor de convolutie lagen berekenen we:
    Conv_2d: Conv2D met 20 filters en kernelgrootte (5, 5):
    •	Ho = (1024 - 5 + 1) / 1 = 1020
    •	Wo = (1024 - 5 + 1) / 1 = 1020
    •	Hi = 1024
    •	Wi = 1024
    •	K = 5
    •	Ci = 1 (grijswaardenafbeelding)
    •	Co = 20
    •	B = 1 (aangezien we slechts één largeTestImage hebben)
    FLOPs_1 = (1020 * 1020 * 1024 * 1024 * 5 * 5 * 1 * 20) * 1
    Conv_2d_1: Conv2D met 20 filters en kernelgrootte (3, 3):
    •	Ho = (1020 - 3 + 1) / 1 = 1018
    •	Wo = (1020 - 3 + 1) / 1 = 1018
    •	Hi = 1020
    •	Wi = 1020
    •	K = 3
    •	Ci = 20
    •	Co = 20
    •	B = 1
    FLOPs_2 = (1018 * 1018 * 1020 * 1020 * 3 * 3 * 20 * 20) * 1

    Conv_2d_2: Conv2D met 10 filters en kernelgrootte (5, 5):
    •	Ho = (1018 - 5 + 1) / 1 = 1014
    •	Wo = (1018 - 5 + 1) / 1 = 1014
    •	Hi = 1018
    •	Wi = 1018
    •	K = 5
    •	Ci = 20
    •	Co = 10
    •	B = 1
    FLOPs_3 = (1014 * 1014 * 1018 * 1018 * 5 * 5 * 20 * 10) * 1
    Het totale aantal FLOPs voor alle convoluties is dan:
    Total FLOPs = FLOPs_1 + FLOPs_2 + FLOPs_3


* Hoe snel zou een grafische kaart met 1 Teraflops dat theoretisch kunnen uitrekenen?
We willen berekenen hoe lang het zou duren om alle berekening uit te voeren die nodig zijn voor de convoluties van onze grote testafbeelding.
Laten we aannemen dat we in totaal 1 miljard berekeningen moeten doen. In dit geval zou het theoretisch slechts 1 seconde duren voor de grafische kaart om al die berekeningen uit te voeren.
Tijd = (1 miljard FLOPs) / (1 Teraflops) = 1 seconde
Maar het is belangrijk om te weten dat deze berekening erg vereenvoudigd is en geen rekening houdt met andere factoren die de prestaties kunnen beinvloeden.


* Waarom zal het in de praktijk wat langer duren?

Dit komt door enkele factoren die de totale berekeningstijd beïnvloeden. Een belangrijke factor is geheugentoegang. Tijdens de berekingen moet de grafische kaart mogelijk gegevens ophalen uit het geheugen, zoals de invoer- en filtergegevens voor de convoluties. Hoewel de kaart zelf ongelooflijk snel kan rekenen, kan het ophalen van gegevens van het geheugen relatief traag zijn. Dit betekent dat er enige vertraging kan optreden terwijl de kaart wacht op de benodigde gegevens. 
Daarnaast is er ook geheugenbandbreedte. Dit verwijst naar de snelheid waarmee gegevens van en naar het geheugen kunnen worden overgedragen. Als de geheugenbandbreedte niet voldoende is om aan de eisen van de berekeningen te voldoen, kan dit resulteren in wachttijden en vertragingen. 
Ook kan natuurlijk de grafische kaart voor meerdere doeleinden tegelijkertijd gebruikt worden, zoals het weergeven van grafische beelden op een scherm. Dus kunnen er ook meerdere taken tegelijkertijd zijn, waardoor het de beschikbare rekenkracht vermindert.

"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(20, kernel_size=(5, 5)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(20, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(10, kernel_size=(5, 5))
        ]
    )
    return model


model = buildMyModel(x_train[0].shape)
model.summary()





"""
## Train the model
"""

batch_size = 4096 # Larger means faster training, but requires more system memory.
epochs = 1000 # for now

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate) #lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
    model.load_weights("myWeights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.

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
print(prediction[0])
print(y_test[0])


# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


print(x_test.shape)

# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

lstWeights=[]
for nLanger in range(len(model.layers)):
    lstWeights.append(model.layers[nLayer].get_weights())

# rebuild model, but with larger input:
model2 = buildMyModel(largeTestImages[0].shape)
model2.summary()

for nLayer in range(len(model2.layers)):
      model2.layers[nLayer].set_weights(model.layers[nLayer].get_weights())

prediction2 = model2.predict(largeTestImages)
baseFilenameForSave='FeaturemapOfLargeTestImage_A_'
ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model2, None, largeTestImages,  0, baseFilenameForSave)

####################
# Drempelwaarde en detectie van posities
threshold = 0.5
digit_positions = []

for digit in range(10):
    digit_position = []

    # Loop door elke featuremap
    for featuremap_index, featuremap in enumerate(prediction2):
        # Loop door elke pixel in de featuremap
        for row in range(featuremap.shape[0]):
            for col in range(featuremap.shape[1]):
                # Controleer of de intensiteitswaarde hoger is dan de drempelwaarde 
                if featuremap[row, col, digit] > threshold:
                    # Voeg de positie toe aan de lisjt
                    digit_position.append((featuremap_index, row, col))
    
    digit_positions.append(digit_position)

# Resultaten weergeven
for digit, positions in enumerate(digit_positions):
    print("Posities van cijfer", digit)
    for position in positions:
        print("Featuremap:", position[0], "Rij:", position[1], "Kolom", position[2])
    print()            
        
####################
