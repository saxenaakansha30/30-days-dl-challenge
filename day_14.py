# Problem: Mini-Project: Build a simple custom CNN-based model using Knowledge Distillation

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy


# Load the ResNet50 pre-trained larger model (Teacher)
teacher_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Add custom classification layers on top
x = teacher_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Final Teacher Model
teacher_model = Model(inputs=teacher_model.inputs, outputs=predictions)

# Compile the teacher model
teacher_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load the dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# Normalize the pixel values between 0-1
X_train = X_train / 255.0
X_val = X_val / 255.0

X_train = tf.image.resize(X_train, (128, 128))
X_val = tf.image.resize(X_val, (128, 128))

# Train the teacher model.
teacher_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1,
)

teacher_output = teacher_model.predict(X_val)

# Define the Student Model

# Define the simple smaller CNN as student model
student_model = Sequential()

student_model.add(Dense(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
student_model.add(MaxPooling2D(3, 3))
student_model.add(Dense(32, (3, 3), activation='relu'))
student_model.add(MaxPooling2D(3, 3))
student_model.add(Dense(64, (3, 3), activation='relu'))
student_model.add(Flatten())
student_model.add(Dense(128, activation='relu'))
student_model.add(Dropout(0.5))
student_model.add(Dense(10, activation='softmax'))

student_model.summary()

# Train the Student Model using Distillation
def distillation_loss(org_prediction, student_prediction, teachers_output, temperature=0.3, alpha=0.5):
    # Calculate the student loss using standard categorical cross entropy
    student_loss = categorical_crossentropy(org_prediction, student_prediction)

    # Student, Teacher soft target distribution
    teachers_soft = K.softmax(teachers_output / temperature)
    student_soft = K.softmax(student_prediction / temperature)

    distillation_loss = K.sum(teachers_soft * K.log(teachers_soft / student_soft))

    # Combined loss: Kullback-Leibler divergence
    return alpha * student_loss + (1 - alpha) * distillation_loss

# Compile the student model
student_model.compile(
    optimizer='adam',
    loss=lambda org_prediction, student_prediction: distillation_loss(org_prediction, student_prediction, teacher_output),
    metrics=['accuracy']
)

# Train the model
student_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)