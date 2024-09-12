import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import utils_general

model = keras.Sequential([
        Dense(512, input_dim=60, activation='elu'),
        Dropout(0.075),            
        Dense(196, activation='elu'),
        Dropout(0.15),
        Dense(32, activation='elu'),
        Dropout(0.35),
        Dense(1)
        ])

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    # Load Config
    config_path = 'config.yaml'
    # Call the function with appropriate parameters
    train_dataset = utils_general.prepare_dataset(decluster_ratio=0.25, threshold=0, susceptibility_ratio=0.5)
    
    # Split the dataset into features (X) and target (y)
    X = train_dataset.iloc[:, 2:-1].values  # Assuming features are from 3rd to 2nd last column
    y = train_dataset.iloc[:, -1].values  # Assuming target is the last column
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    utils_general.save_split(X_train,X_test,y_train,y_test)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[utils_general.coeff_determination])
    epch = 960
    history = model.fit(X_train, y_train, batch_size=1024, epochs=epch, verbose=1, validation_split=0.1)
    
    model.save_weights('data/training_results/weights.hdf5')
    
    TEST_Pred = model.predict(X_test)
    TRAIN_Pred = model.predict(X_train)
    
    # Performance metrics
    utils_general.plot_scat(y_train, TRAIN_Pred, alpha=0.5)
    utils_general.plot_scat(y_test, TEST_Pred, alpha=0.5)
    utils_general.evaluate(y_test, TEST_Pred)