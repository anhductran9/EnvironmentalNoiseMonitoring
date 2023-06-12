from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import load_model
import extract_features
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from keras import backend as K
from matplotlib import pyplot

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
    model.add(Conv1D(32, 2, activation='relu', input_shape=(40, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    return model

#def create_cnn(num_labels):

    #model = Sequential()
    #model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
    #model.add(Conv1D(64, 3, activation='relu'))
    #model.add(MaxPooling1D(3))
    #model.add(Conv1D(128, 3, activation='relu'))
    #model.add(Conv1D(128, 3, activation='relu'))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.5))
    #model.add(Dense(num_labels))
    #model.add(Activation('softmax'))
    #return model

def train(model,X_train, X_test, y_train, y_test,model_file):    
    
    # compile the model 
    model.compile(loss = 'categorical_crossentropy',metrics=['accuracy', f1],optimizer='adam')
    print(model.summary())
    print("training for 100 epochs with batch size 32")
    history = model.fit(X_train,y_train,batch_size= 32, epochs = 100, validation_data=(X_test,y_test))
    
    # save model to disk
    print("Saving model to disk")
    model.save(model_file)

    # plot
        
    # plot accuracy during training
    pyplot.subplot(221)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    #pyplot.show()

    # plot f1 score during training
    pyplot.subplot(222)
    pyplot.title('F1 Score')
    pyplot.plot(history.history['f1'], label='train')
    pyplot.plot(history.history['val_f1'], label='test')
    pyplot.legend()
    #pyplot.show()

    # plot loss during training
    pyplot.subplot(212)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    
    pyplot.show()

def compute(X_test,y_test,model_file):

    # load model from disk
    loaded_model = load_model(model_file, custom_objects={"f1": f1})
    score = loaded_model.evaluate(X_test,y_test)
    return score[0],score[1]*100



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        # Recall metric.
        # Only computes a batch-wise average of recall.
        # Computes the recall, a metric for multi-class classification of how many relevant items are selected.

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        # Precision metric.
        # Only computes a batch-wise average of precision.
        # Computes the precision, a metric for multi-class classification of how many selected items are relevant.

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))




def predict(filename,le,model_file):

    model = load_model(model_file, custom_objects={"f1": f1})
    prediction_feature = extract_features.get_features(filename)
    sound_level = extract_features.get_db(filename)
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
        sound_name = "Dog"
    elif (sound_number == 6):
        sound_name = "Rooster"
    elif (sound_number == 7):
        sound_name = "Fire"
    elif (sound_number == 8):
        sound_name = "Person talking"
    elif (sound_number == 9):
        sound_name = "Gunshot"
    else: sound_name = ""

    print("Predicted class : ", sound_name, "; Predicted decibels : ",sound_level)
    for i in range(len(predicted_proba)):
        
        # When predicting
        #category = LabelEncoder().inverse_transform(np.array([i]))
        #le.fit(predicted_proba)
        
        # When building a model
        category = le.inverse_transform(np.array([i]))
        
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        
    with open("output.csv", "w", newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["ID", "SoundID","Sound Classification", "Decibels"])
        writer.writerow(["", "", sound_name, sound_level])