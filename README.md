# Lunar Lander GYM dengan PyTorch

Proyek ini merupakan implementasi dari simulasi Lunar Lander menggunakan library OpenAI Gym dan framework PyTorch. Proyek ini berfokus pada pengembangan agen kecerdasan buatan yang dapat mendaratkan roket dengan aman di permukaan bulan menggunakan teknik reinforcement learning.

## Demo Video

Berikut adalah demo singkat dari simulasi:

### Demo tanpa LSTM (Pytorch)

![Lunar Lander GYM Demo](./demo.gif)

### Demo dengan LSTM (Tensorflow)

![Lunar Lander LSTM GYM Demo](./demo_dengan_LSTM.gif)

di pembaharuan ini saya menambah kan model dengan LSTM sehinga model dapat belajar dari urutan kejadian sebelum nya untuk yang LSTM ini menggunakan Tensorflow
history:
![Lunar Lander GYM reward](./reward_history.png)

## Fitur

- **Simulasi OpenAI Gym** untuk Lunar Lander.
- **Model berbasis PyTorch** untuk pembelajaran reinforcement.
- **Agen cerdas** yang belajar dari pengalaman untuk mendaratkan roket dengan aman.

## Instalasi

Untuk menjalankan proyek ini, ikuti langkah-langkah berikut:

1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/LunarLanderGymPyTorch.git
   ```
