"""
3_train_model.py
Train deep learning model for skin cancer detection
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime

class SkinCancerModel:
    def __init__(self, img_size=224, learning_rate=0.001):
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build model using Transfer Learning with MobileNetV2"""
        
        print("Building model...")
        
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        
        x = base_model(x, training=False)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("Model built successfully")
        print("\nModel Summary:")
        self.model.summary()
        
        return self.model
    
    def create_data_generators(self, batch_size=32):
        """Create data generators for training"""
        
        print("\nCreating data generators...")
        
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        train_generator = train_datagen.flow_from_directory(
            'dataset/train',
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            'dataset/val',
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Train samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def get_callbacks(self):
        """Create callbacks for training"""
        
        os.makedirs('models', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                f'models/best_model_{timestamp}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, epochs=20, batch_size=32):
        """Train the model"""
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        train_gen, val_gen = self.create_data_generators(batch_size)
        
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks()
        
        print(f"\nTraining for {epochs} epochs...")
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def plot_history(self):
        """Plot training history"""
        
        if self.history is None:
            print("No training history available!")
            return
        
        print("\nPlotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Saved training_history.png")
        plt.close()
    
    def save_model(self, filename='final_model.keras'):
        """Save final model"""
        
        if self.model is None:
            print("No model to save!")
            return
        
        os.makedirs('models', exist_ok=True)
        filepath = f'models/{filename}'
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

def main():
    print("="*60)
    print("SKIN CANCER DETECTION MODEL TRAINING")
    print("="*60)
    
    if not os.path.exists('dataset/train'):
        print("ERROR: dataset/train not found!")
        print("Run: python 1_organize_dataset.py first")
        return
    
    model = SkinCancerModel(img_size=224, learning_rate=0.001)
    
    model.build_model()
    
    model.train(epochs=20, batch_size=32)
    
    model.plot_history()
    
    model.save_model('skin_cancer_model_final.keras')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nNext step: python 4_evaluate_model.py")

if __name__ == "__main__":
    main()