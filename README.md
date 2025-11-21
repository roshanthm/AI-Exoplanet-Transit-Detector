# 🌌 AI Exoplanet Transit Detector

An interactive, educational web application built with **Streamlit** and **TensorFlow/Keras** for simulating and detecting exoplanetary transit signals in noisy light curves.

This project demonstrates how a **1D Convolutional Neural Network (CNN)** can be trained to perform signal detection, a task traditionally handled by complex statistical models.

## ✨ Features

* **Interactive Simulation:** Adjust parameters like **Transit Depth**, **Duration**, and **Noise Level** to generate unique light curves.
* **AI Detection:** A trained CNN instantly analyzes the generated light curve and provides a **Probability** of a transit being present.
* **Data Visualization:** Real-time plotting of the light curve using Matplotlib.
* **Custom UI:** Features a custom HTML/CSS-injected **video background** for an immersive, space-themed experience.

## 💻 Tech Stack

* **Frontend/App Framework:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [TensorFlow 2.x](https://www.tensorflow.org/) & [Keras](https://keras.io/)
* **Data Processing:** [NumPy](https://numpy.org/)
* **Visualization:** [Matplotlib](https://matplotlib.org/)

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ installed. All required packages can be installed via `pip`.

```bash
pip install streamlit tensorflow keras numpy matplotlib
