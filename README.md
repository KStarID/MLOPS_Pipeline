# Submission 1: Apple Quality Classification

Nama: Kemal Aziz

Username dicoding: kstarid

|                         | Deskripsi                                                                                                                                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)                                                                                                                                                                        |
| Masalah                 | Mengklasifikasikan kualitas apel (baik/buruk) berdasarkan karakteristik fisik dan sensorik seperti ukuran, berat, tingkat kemanisan, kerenyahan, kejusan, kematangan, dan keasaman.                                                                            |
| Solusi machine learning | Mengimplementasikan pipeline machine learning menggunakan TensorFlow Extended (TFX) untuk membangun model klasifikasi biner yang dapat memprediksi kualitas apel.                                                                                              |
| Metode pengolahan       | Data dinormalisasi menggunakan z-score normalization pada fitur numerik. Label 'Quality' dikonversi menjadi nilai biner (0 untuk "bad", 1 untuk "good"). Pipeline TFX digunakan untuk memastikan konsistensi transformasi antara tahap training dan inferensi. |
| Arsitektur model        | Model neural network dengan input layer untuk setiap fitur numerik, layer concatenate, BatchNormalization, beberapa hidden layer dengan jumlah dan ukuran sesuai hasil tuning, dan output layer dengan aktivasi sigmoid untuk klasifikasi biner.               |
| Metrik evaluasi         | Binary Accuracy, AUC (Area Under Curve), dan ExampleCount. Model dianggap "blessed" jika accuracy di atas 0.8 (80%).                                                                                                                                           |
| Performa model          | Model mencapai accuracy sekitar 95% pada data evaluasi, menunjukkan kemampuan yang baik dalam membedakan apel berkualitas baik dan buruk.                                                                                                                      |

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
