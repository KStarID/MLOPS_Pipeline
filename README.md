# Apple Quality Classification Model

## Deskripsi Proyek

Proyek ini mengimplementasikan pipeline machine learning menggunakan TensorFlow Extended (TFX) untuk memprediksi kualitas apel (good/bad) berdasarkan fitur-fitur seperti ukuran, berat, tingkat kemanisan, kerenyahan, kejusan, kematangan, dan keasaman.

## Dataset

Dataset yang digunakan adalah Apple Quality dataset yang berisi informasi tentang berbagai karakteristik apel dan klasifikasi kualitasnya.

Fitur-fitur yang digunakan:

- A_id: ID apel
- Size: Ukuran apel
- Weight: Berat apel
- Sweetness: Tingkat kemanisan
- Crunchiness: Tingkat kerenyahan
- Juiciness: Tingkat kejusan
- Ripeness: Tingkat kematangan
- Acidity: Tingkat keasaman

Label:

- Quality: Kualitas apel (good/bad)

## Komponen Pipeline TFX

Pipeline machine learning yang diimplementasikan terdiri dari komponen-komponen berikut:

1. **ExampleGen**: Mengimpor data dari CSV dan membaginya menjadi data training dan evaluasi
2. **StatisticsGen**: Menganalisis statistik dari data
3. **SchemaGen**: Membuat schema berdasarkan statistik data
4. **ExampleValidator**: Memvalidasi data berdasarkan schema
5. **Transform**: Melakukan preprocessing data (normalisasi, konversi tipe data)
6. **Tuner**: Mencari hyperparameter terbaik untuk model
7. **Trainer**: Melatih model dengan hyperparameter terbaik
8. **Resolver**: Menyelesaikan model terbaik
9. **Evaluator**: Mengevaluasi performa model
10. **Pusher**: Menyimpan model yang sudah dilatih

## Model Architecture

Model yang digunakan adalah neural network dengan arsitektur sebagai berikut:

- Input layer untuk setiap fitur numerik
- Concatenate layer untuk menggabungkan semua fitur
- BatchNormalization layer
- Hidden layers dengan jumlah dan ukuran sesuai hasil tuning
- Output layer dengan aktivasi sigmoid untuk klasifikasi biner

## Evaluasi Model

Model dievaluasi menggunakan metrik:

- Accuracy
- Precision
- Recall

## Deployment dengan TensorFlow Serving

### Langkah-langkah Deployment

1. **Build Docker Image**

   ```bash
   docker build -t apple-quality-serving .
   ```

2. **Jalankan Container**

   ```bash
   docker run -p 8501:8501 -p 8500:8500 apple-quality-serving
   ```

3. **Akses REST API**
   Model dapat diakses melalui REST API di:
   ```
   http://localhost:8501/v1/models/apple_quality_model:predict
   ```

### Contoh Request

```json
{
  "instances": [
    {
      "A_id_xf": [0.5],
      "Size_xf": [0.5],
      "Weight_xf": [0.6],
      "Sweetness_xf": [0.7],
      "Crunchiness_xf": [0.8],
      "Juiciness_xf": [0.9],
      "Ripeness_xf": [0.7],
      "Acidity_xf": [0.5]
    }
  ]
}
```

### Testing Model

Untuk menguji model yang sudah di-deploy, jalankan notebook `testing.ipynb` yang telah disediakan.
