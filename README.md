# Submission 1: Apple Quality Classification

Nama: Kemal Aziz

Username dicoding: kstarid

|                         | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Apple Quality Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Masalah                 | Kualitas apel merupakan faktor penting dalam industri pertanian dan pangan yang mempengaruhi nilai jual dan penerimaan konsumen. Namun, penilaian kualitas apel secara manual membutuhkan waktu, tenaga, dan keahlian khusus yang tidak selalu tersedia. Berbagai karakteristik fisik dan sensorik seperti ukuran, berat, tingkat kemanisan, kerenyahan, kejusan, kematangan, dan keasaman berperan dalam menentukan kualitas apel, tetapi hubungan kompleks antar karakteristik ini sulit dianalisis secara manual. Oleh karena itu, diperlukan sebuah sistem otomatis yang dapat mengklasifikasikan kualitas apel (baik/buruk) berdasarkan karakteristik-karakteristik tersebut secara akurat dan konsisten. |
| Solusi machine learning | Solusi yang diimplementasikan adalah membangun pipeline machine learning end-to-end menggunakan TensorFlow Extended (TFX). Pipeline ini mengotomatisasi seluruh proses mulai dari penyerapan data CSV, validasi data, transformasi fitur, pencarian hyperparameter optimal melalui tuning, pelatihan model neural network, evaluasi performa, hingga penyiapan model untuk deployment menggunakan TensorFlow Serving. Pendekatan ini memastikan konsistensi dan reprodusibilitas dalam seluruh alur kerja machine learning, serta memudahkan transisi dari pengembangan ke produksi.                                                                                                                           |
| Metode pengolahan       | Data diolah menggunakan komponen TFX Transform dengan fungsi preprocessing_fn dalam apple_transform.py. Metode pengolahan yang diterapkan meliputi:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

1. Normalisasi fitur numerik menggunakan z-score normalization (tft.scale_to_z_score) untuk semua fitur numerik: A_id, Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, dan Acidity.
2. Konversi label kategorik 'Quality' menjadi nilai biner (0 untuk "bad", 1 untuk "good") menggunakan tf.cast dan tf.equal.
3. Pipeline TFX digunakan untuk memastikan konsistensi transformasi antara tahap training dan inferensi, sehingga menghindari training-serving skew. |
   | Arsitektur model | Arsitektur model final ditentukan melalui proses hyperparameter tuning. Berdasarkan hasil tuning, arsitektur terbaik yang ditemukan adalah:
   • Input Layer: Satu input layer untuk setiap fitur numerik (8 fitur).
   • Concatenate Layer: Menggabungkan semua input fitur.
   • BatchNormalization Layer: Menormalkan aktivasi dari layer sebelumnya.
   • Hidden Layer 1: Dense dengan 128 unit dan aktivasi ReLU.
   • Hidden Layer 2: Dense dengan 160 unit dan aktivasi ReLU.
   • Dropout Layer: Dengan rate 0.2 untuk mencegah overfitting.
   • Output Layer: Dense dengan 1 unit dan aktivasi sigmoid untuk klasifikasi biner.
   • Optimizer: Adam dengan learning rate 0.001.
   • Loss Function: Binary Crossentropy. |
   | Metrik evaluasi | Metrik utama yang digunakan untuk mengevaluasi model adalah:
4. Binary Accuracy: Mengukur proporsi prediksi yang benar (baik positif maupun negatif).
5. AUC (Area Under the ROC Curve): Mengukur kemampuan model untuk membedakan antara kelas positif dan negatif.
6. ExampleCount: Menghitung jumlah contoh yang dievaluasi.

Model dianggap "blessed" (layak untuk deployment) jika mencapai binary accuracy minimal 0.8 (80%) pada data evaluasi. Threshold ini dipilih untuk memastikan model memiliki tingkat akurasi yang cukup tinggi sebelum digunakan dalam produksi. |
| Performa model | Model mencapai performa yang sangat baik dengan:
• Binary Accuracy: sekitar 95% pada data evaluasi
• AUC: di atas 0.95, menunjukkan kemampuan diskriminatif yang sangat baik
• Hasil tuning menunjukkan bahwa arsitektur terbaik dengan 2 hidden layer (128 dan 160 unit) dan dropout 0.2 menghasilkan performa optimal.

Performa ini menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam mengklasifikasikan kualitas apel berdasarkan karakteristik fisik dan sensoriknya, dengan tingkat kesalahan yang relatif rendah. |

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
