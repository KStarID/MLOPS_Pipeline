# Submission 1: Apple Quality Classification

Nama: Kemal Aziz

Username dicoding: kstarid

|                         | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) <br><br>Dataset ini berisi 4000 sampel apel dengan berbagai karakteristik fisik dan sensorik. Setiap sampel memiliki 8 fitur: ID apel, ukuran, berat, tingkat kemanisan, kerenyahan, kejusan, kematangan, dan keasaman. Target klasifikasi adalah kualitas apel yang dikategorikan sebagai "good" (baik) atau "bad" (buruk). Dataset ini memungkinkan pemodelan hubungan antara karakteristik fisik dan sensorik apel dengan kualitasnya secara keseluruhan.                                                                                                                                                           |
| Masalah                 | Kualitas apel merupakan faktor penting dalam industri pertanian dan pangan yang mempengaruhi nilai jual dan penerimaan konsumen. Namun, penilaian kualitas apel secara manual membutuhkan waktu, tenaga, dan keahlian khusus yang tidak selalu tersedia. Berbagai karakteristik fisik dan sensorik seperti ukuran, berat, tingkat kemanisan, kerenyahan, kejusan, kematangan, dan keasaman berperan dalam menentukan kualitas apel, tetapi hubungan kompleks antar karakteristik ini sulit dianalisis secara manual. Oleh karena itu, diperlukan sebuah sistem otomatis yang dapat mengklasifikasikan kualitas apel (baik/buruk) berdasarkan karakteristik-karakteristik tersebut secara akurat dan konsisten. |
| Solusi machine learning | Solusi yang diimplementasikan adalah membangun pipeline machine learning end-to-end menggunakan TensorFlow Extended (TFX). Pipeline ini mengotomatisasi seluruh proses mulai dari penyerapan data CSV, validasi data, transformasi fitur, pencarian hyperparameter optimal melalui tuning, pelatihan model neural network, evaluasi performa, hingga penyiapan model untuk deployment menggunakan TensorFlow Serving. Pendekatan ini memastikan konsistensi dan reprodusibilitas dalam seluruh alur kerja machine learning, serta memudahkan transisi dari pengembangan ke produksi.                                                                                                                           |
| Metode pengolahan       | Data dinormalisasi menggunakan z-score normalization pada fitur numerik. Label 'Quality' dikonversi menjadi nilai biner (0 untuk "bad", 1 untuk "good"). Pipeline TFX digunakan untuk memastikan konsistensi transformasi antara tahap training dan inferensi.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Arsitektur model        | Model neural network dengan input layer untuk setiap fitur numerik, layer concatenate, BatchNormalization, beberapa hidden layer dengan jumlah dan ukuran sesuai hasil tuning, dan output layer dengan aktivasi sigmoid untuk klasifikasi biner.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Metrik evaluasi         | Binary Accuracy, AUC (Area Under Curve), dan ExampleCount. Model dianggap "blessed" jika accuracy di atas 0.8 (80%).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Performa model          | Model mencapai accuracy sekitar 95% pada data evaluasi, menunjukkan kemampuan yang baik dalam membedakan apel berkualitas baik dan buruk.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## Komponen Pipeline TFX

Pipeline machine learning yang diimplementasikan terdiri dari komponen-komponen berikut:

1. **ExampleGen**: Mengimpor data dari CSV dan membaginya menjadi data training (80%) dan evaluasi (20%)
2. **StatisticsGen**: Menganalisis statistik dari data seperti distribusi, nilai rata-rata, dan standar deviasi
3. **SchemaGen**: Membuat schema berdasarkan statistik data untuk mendefinisikan tipe dan batasan fitur
4. **ExampleValidator**: Memvalidasi data berdasarkan schema untuk mendeteksi anomali
5. **Transform**: Melakukan preprocessing data (normalisasi z-score dan konversi label)
6. **Tuner**: Mencari hyperparameter terbaik untuk model neural network
7. **Trainer**: Melatih model dengan hyperparameter terbaik dari hasil tuning
8. **Resolver**: Menyelesaikan model terbaik untuk evaluasi dan deployment
9. **Evaluator**: Mengevaluasi performa model dengan metrik binary accuracy dan AUC
10. **Pusher**: Menyimpan model yang sudah dilatih dan "blessed" ke lokasi deployment

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

### Testing Model

Untuk menguji model yang sudah di-deploy, jalankan notebook `testing.ipynb` yang telah disediakan.
