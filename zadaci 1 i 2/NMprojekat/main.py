from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import keras_tuner as kt
from keras import models
from keras import layers

from keras import optimizers
from keras.regularizers import l2


###########################     PRVI DEO    ###############################################

genre = pd.read_csv(r'C:\Users\PC\Desktop\nm\nm projekat\Genres.csv')

redosled = [0, 1, 2]

genre['genre'] = genre['genre'].map({'Rap': 0, 'Pop': 1, 'RnB': 2})

ulaz = genre[['danceability', 'energy', 'key', 'loudness', 'mode',
             'speechiness', 'acousticness', 'instrumentalness',
             'liveness', 'valence', 'tempo']]

izlaz = genre['genre']

print(izlaz.shape)

instance = genre['genre'].value_counts().reindex(redosled)

plt.figure(figsize=(7,5))
slika = instance.plot(kind='bar', color=['green', 'red', 'lightblue'])
plt.title('Muzički žanrovi')
plt.xlabel('Žanr')
plt.ylabel('Ukupan broj instanci')
slika.set_xticklabels(['Rap', 'Pop', 'RnB'], rotation=0)
plt.tight_layout()
plt.show()

###########################     DRUGI DEO    ###############################################

# TRENING TEST I VALIDACIJA KLASE

#ulaz_trening, ulaz_validacija, izlaz_trening, izlaz_validacija = train_test_split(ulaz_trening, izlaz_trening, train_size=0.8, random_state=45)

ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(
    ulaz, izlaz, test_size=0.2, random_state=45, stratify=izlaz)
ulaz_trening, ulaz_validacija, izlaz_trening, izlaz_validacija = train_test_split(
    ulaz_trening, izlaz_trening, train_size=0.8, stratify=izlaz_trening)


# NORMALIZACIJA
scaler = StandardScaler().fit(ulaz_trening)
ulaz_trening_norm = scaler.transform(ulaz_trening)
ulaz_validacija_norm = scaler.transform(ulaz_validacija)
ulaz_test_norm = scaler.transform(ulaz_test)

smote = SMOTE(random_state=45)
ulaz_trening_bal, izlaz_trening_bal = smote.fit_resample(ulaz_trening_norm, izlaz_trening)


ulaz_trening_bal = ulaz_trening_bal.astype(np.float32)
izlaz_trening_bal = izlaz_trening_bal.astype(np.int32)
ulaz_validacija_norm = ulaz_validacija_norm.astype(np.float32)
izlaz_validacija = izlaz_validacija.astype(np.int32)
ulaz_test_norm = ulaz_test_norm.astype(np.float32)
izlaz_test = izlaz_test.astype(np.int32)

# RASPODELA ODBIRAKA PO SKUPOVIMA ZA TRENING TEST I VALIDACIJU

counts = [
    (np.count_nonzero(izlaz_trening == 0),
     np.count_nonzero(izlaz_trening == 1),
     np.count_nonzero(izlaz_trening == 2)),

    (np.count_nonzero(izlaz_validacija == 0),
     np.count_nonzero(izlaz_validacija == 1),
     np.count_nonzero(izlaz_validacija == 2)),

    (np.count_nonzero(izlaz_test == 0),
     np.count_nonzero(izlaz_test == 1),
     np.count_nonzero(izlaz_test == 2))
]

labels = ['trening', 'validacija', 'test']
rap = [c[0] for c in counts]
pop = [c[1] for c in counts]
rnb = [c[2] for c in counts]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
ax.bar(x - width, rap, width, label='Rap')
ax.bar(x, pop, width, label='Pop')
ax.bar(x + width, rnb, width, label='RnB')

ax.set_ylabel('Broj odbiraka')
ax.set_title('Raspodela odbiraka po skupovima')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


###########################   HIPERPARAMETRI    ###############################################

def make_model(hp):
    model = models.Sequential()
    model.add(layers.Input((ulaz_trening_norm.shape[1],)))

    no_units = hp.Int('units', min_value=32, max_value=256, step=16)
    act = hp.Choice('activation', values=['sigmoid', 'relu', 'tanh'])

    reg = hp.Float('reg', 0.001, 0.5, 0.005)

    drop = hp.Float('drop', 0, 0.8, 0.05)

    lr = hp.Float('learning_rate', 1e-5, 1e-2, 1e-4)

    model.add(layers.Dense(units=no_units, activation=act))
    model.add(layers.Dense(no_units // 2, activation=act, kernel_regularizer=l2(reg)))

    model.add(layers.Dense(16, activation='relu', kernel_regularizer=l2(reg)))
    model.add(layers.Dropout(drop))
    model.add(layers.Dense(3, activation='softmax')) #zbog klasa 0, 1, 2

    opt = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

stop_early = EarlyStopping(monitor='val_accuracy',
                        mode='max',
                        patience=5,
                        restore_best_weights=True,
                        verbose=2)


tuner = kt.RandomSearch(make_model,
                        objective='val_accuracy',
                        overwrite=True,
                        max_trials=30)

tuner.search(ulaz_trening_bal, izlaz_trening_bal,
             epochs=50,
             batch_size=64,
             validation_data=(ulaz_validacija_norm, izlaz_validacija),
             callbacks=[stop_early],
             verbose=3)


###########            NAJBOLJI MODEL              ###############
best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]

print('Optimalan broj neurona u prvom skrivenom sloju: ', best_hyperparam['units'])
print('Optimalna funkcija aktivacije u prvom skrivenom sloju: ', best_hyperparam['activation'])
print('Optimalan koeficijent regularizacije u drugom skrivenom sloju: ', best_hyperparam['reg'])
print('Optimalan dropout rate: ', best_hyperparam['drop'])
print('Optimalna konstanta obučavanja: ', best_hyperparam['learning_rate'])




best_model = tuner.hypermodel.build(best_hyperparam)

history = best_model.fit(
    ulaz_trening_bal,
    izlaz_trening_bal,
    epochs=50,
    batch_size=64,
    validation_data=(ulaz_validacija_norm, izlaz_validacija),
    callbacks=[stop_early],
    verbose=0
)

loss, acc = best_model.evaluate(ulaz_validacija_norm, izlaz_validacija, verbose=0)
print(f"Najbolja tačnost na validacionom skupu: {acc:.4f}")


plt.figure()
plt.plot(history.history['loss'], label='Loss trening')
plt.plot(history.history['val_loss'], label='Loss validacija')
plt.xlabel('Epohe')
plt.ylabel('Vrednost loss funkcije')
plt.title('Loss funkcija: trening i validacioni skup')
plt.legend()
plt.grid(True)
plt.show()

y_pred = np.argmax(best_model.predict(ulaz_trening_bal, verbose=0), axis=1)
cm = confusion_matrix(izlaz_trening_bal, y_pred)

labels = ['Rap', 'Pop', 'RnB']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica konfuzije za trening skup")
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()


model = RandomForestClassifier(n_estimators=200, random_state=45)
model.fit(ulaz_trening_bal, izlaz_trening_bal)

y_pred = model.predict(ulaz_test_norm)
y_true = izlaz_test

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
sens = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Tacnost na test skupu: {acc*100:.2f} %")
print(f"Preciznost na test skupu: {prec*100:.2f} %")
print(f"Osetljivost na test skupu: {sens*100:.2f} %")
print(f"F1-score na test skupu: {f1*100:.2f} %")

cm = confusion_matrix(y_true, y_pred)
labels = ['Rap', 'Pop', 'RnB']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica konfuzije za test skup")
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()