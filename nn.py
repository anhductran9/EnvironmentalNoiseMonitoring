from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import load_model
import extract_features
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv

def create_mlp(num_labels):

    model = Sequential()
    model.add(Dense(256,input_shape = (40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256,input_shape = (40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    return model

def create_cnn(num_labels):

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    return model

def train(model,X_train, X_test, y_train, y_test,model_file):    
    
    # compile the model 
    model.compile(loss = 'categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    print(model.summary())

    print("training for 100 epochs with batch size 32")
   
    model.fit(X_train,y_train,batch_size= 32, epochs = 100, validation_data=(X_test,y_test))
    
    # save model to disk
    print("Saving model to disk")
    model.save(model_file)

def compute(X_test,y_test,model_file):

    # load model from disk
    loaded_model = load_model(model_file)
    score = loaded_model.evaluate(X_test,y_test)
    return score[0],score[1]*100

def predict(filename,le,model_file):

    model = load_model(model_file)
    prediction_feature = extract_features.get_features(filename)
    if model_file == "trained_mlp.h5":
        prediction_feature = np.array([prediction_feature])
    elif model_file == "trained_cnn.h5":    
        prediction_feature = np.expand_dims(np.array([prediction_feature]),axis=2)

    predicted_vector = model.predict(prediction_feature)
    predicted_proba = predicted_vector[0]
    
    sound_number = np.argmax(predicted_proba)
    if (sound_number == 0):
        sound_name = "Vehicle"
    elif (sound_number == 1):
        sound_name = "Rain"
    elif (sound_number == 2):
        sound_name = "Plane"
    elif (sound_number == 3):
        sound_name = "Karaoke"
    elif (sound_number == 4):
        sound_name = "Construction"
    elif (sound_number == 5):
        sound_name = "Machine"
    elif (sound_number == 6):
        sound_name = "Rooster"
    elif (sound_number == 7):
        sound_name = "Fire"
    else: sound_name = ""

    #print("Predicted class : ", np.argmax(predicted_proba))
    print("Predicted class : ", sound_name)
    for i in range(len(predicted_proba)):
        #category = LabelEncoder().inverse_transform(np.array([i]))
        le.fit(predicted_proba)
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        
    with open("submission.csv", "w", newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["Sound Classification"])
        writer.writerow([sound_name])