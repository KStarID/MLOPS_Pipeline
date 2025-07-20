import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers, callbacks
from tfx.components.trainer.fn_args_utils import FnArgs
from apple_transform import NUMERIC_FEATURES, LABEL_KEY, transformed_name
import logging

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Membuat dataset dari file TFRecord hasil transform."""
    try:
        transformed_feature_spec = tf_transform_output.transformed_feature_spec()
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type="GZIP"),
            label_key=transformed_name(LABEL_KEY)
        )
        return dataset
    except Exception as e:
        logging.error(f"Error dalam _input_fn: {e}")
        raise

def _get_hyperparameters_with_defaults(hparams_dict):
    """Menambahkan nilai default untuk hyperparameter yang mungkin hilang."""
    defaults = {
        'num_layers': 2,
        'units_0': 64,
        'units_1': 32,
        'units_2': 16,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    
    # Jika hparams_dict None atau kosong, gunakan defaults
    if not hparams_dict:
        return defaults
    
    # Gabungkan nilai default dengan nilai yang ada
    result = defaults.copy()
    result.update(hparams_dict)
    return result

def _build_keras_model(hparams):
    """Membangun model Keras berdasarkan hyperparameter."""
    try:
        # Pastikan semua hyperparameter yang diperlukan tersedia
        hparams = _get_hyperparameters_with_defaults(hparams)
        
        # Buat input untuk semua fitur numerik
        inputs = {}
        for feature_name in NUMERIC_FEATURES:
            transformed_name_key = transformed_name(feature_name)
            inputs[transformed_name_key] = layers.Input(
                shape=(1,), name=transformed_name_key, dtype=tf.float32
            )
        
        # Gabungkan semua fitur
        if len(inputs) > 1:
            x = layers.concatenate(list(inputs.values()))
        else:
            x = list(inputs.values())[0]
        
        # Tambahkan normalisasi batch
        x = layers.BatchNormalization()(x)
        
        # Tambahkan hidden layers
        num_layers = hparams['num_layers']
        for i in range(num_layers):
            units_key = f'units_{i}'
            if units_key in hparams:
                units = hparams[units_key]
            else:
                units = 32  # Default jika tidak ada
            
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(hparams['dropout'])(x)
        
        # Output layer untuk klasifikasi biner
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Buat dan compile model
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hparams['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        # Tampilkan ringkasan model
        model.summary()
        
        return model
    except Exception as e:
        logging.error(f"Error dalam _build_keras_model: {e}")
        raise

def run_fn(fn_args: FnArgs):
    """Fungsi utama untuk melatih model."""
    try:
        logging.info("Memulai proses training...")
        
        # Load transform output
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
        
        # Buat dataset untuk training dan evaluasi
        train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
        eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)
        
        # Ambil hyperparameter dari hasil tuning
        if hasattr(fn_args, 'hyperparameters') and fn_args.hyperparameters:
            if isinstance(fn_args.hyperparameters, dict) and 'values' in fn_args.hyperparameters:
                hparams = fn_args.hyperparameters['values']
            else:
                logging.warning("Format hyperparameters tidak sesuai, menggunakan default")
                hparams = {}
        else:
            logging.warning("Hyperparameters tidak tersedia, menggunakan default")
            hparams = {}
        
        # Buat model
        model = _build_keras_model(hparams)
        
        # Callbacks untuk training
        training_callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3
            ),
            callbacks.TensorBoard(
                log_dir=fn_args.model_run_dir,
                update_freq='epoch'
            )
        ]
        
        # Latih model
        logging.info("Mulai training model...")
        history = model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            epochs=50,
            callbacks=training_callbacks
        )
        
        # Simpan model
        logging.info(f"Menyimpan model ke {fn_args.serving_model_dir}")
        model.save(fn_args.serving_model_dir, save_format='tf')
        
        # Evaluasi model dan simpan metrik
        logging.info("Mengevaluasi model...")
        # Tambahkan parameter steps ke evaluate untuk menghindari error infinite dataset
        eval_result = model.evaluate(
            eval_dataset, 
            steps=fn_args.eval_steps  # Tambahkan parameter steps
        )
        
        # Simpan metrik evaluasi
        metrics_path = f"{fn_args.serving_model_dir}/metrics.txt"
        logging.info(f"Menyimpan metrik evaluasi ke {metrics_path}")
        with open(metrics_path, 'w') as f:
            f.write(f'Accuracy: {eval_result[1]}\n')
            f.write(f'Precision: {eval_result[2]}\n')
            f.write(f'Recall: {eval_result[3]}\n')
        
        logging.info("Training selesai!")
        
    except Exception as e:
        logging.error(f"Error dalam run_fn: {e}")
        raise