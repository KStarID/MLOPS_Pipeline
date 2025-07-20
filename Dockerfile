FROM tensorflow/serving:2.8.0

# Copy model ke direktori model di container
COPY pipelines/kemal-aziz-pipeline/Trainer/model/50/Format-Serving /models/apple_quality_model

# Set environment variable
ENV MODEL_NAME=apple_quality_model

# Expose port
EXPOSE 8501
EXPOSE 8500

# Start TensorFlow Serving (perbaiki format perintah)
CMD tensorflow_model_server --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=/models/${MODEL_NAME} 