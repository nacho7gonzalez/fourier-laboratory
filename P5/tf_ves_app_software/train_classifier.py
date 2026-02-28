# ---------- train_classifier.py ----------
"""
Script para entrenar el RandomForest para HAR (6 clases) y guardar el modelo + scaler.
Uso: python train_classifier.py
Resultado: models/rf_har_model.joblib  (contiene dict {'model':..., 'scaler':...})
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, butter, filtfilt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------------------------------------------------------------
# Función de extracción de características (adaptada)
# ------------------------------------------------------------------

def calcular_caracteristicas_avanzadas_por_ventanas(df, fs=50, window_s=2.56, overlap=0.5):
    """
    Calcula características en ventanas solapadas. Se asume que el dataframe tiene
    columnas: ['body_acc_x','body_acc_y','body_acc_z','gyro_x','gyro_y','gyro_z','label']
    """
    n = len(df)
    window_samples = int(window_s * fs)
    step = int(window_samples * (1 - overlap))
    features = []
    labels_window = []

    for start in range(0, n - window_samples + 1, step):
        end = start + window_samples
        x_acc = df['body_acc_x'].iloc[start:end].values
        y_acc = df['body_acc_y'].iloc[start:end].values
        z_acc = df['body_acc_z'].iloc[start:end].values
        x_gyro = df['gyro_x'].iloc[start:end].values
        y_gyro = df['gyro_y'].iloc[start:end].values
        z_gyro = df['gyro_z'].iloc[start:end].values
        label_win = df['label'].iloc[start:end].values if 'label' in df.columns else np.zeros(window_samples, dtype=int)

        # Estadísticos tiempo por eje
        mean_x = np.mean(x_acc)
        mean_y = np.mean(y_acc)
        mean_z = np.mean(z_acc)
        std_x = np.std(x_acc)
        std_y = np.std(y_acc)
        std_z = np.std(z_acc)
        max_x = np.max(x_acc)
        max_y = np.max(y_acc)
        max_z = np.max(z_acc)
        min_x = np.min(x_acc)
        min_y = np.min(y_acc)
        min_z = np.min(z_acc)
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        median_x = np.median(x_acc)
        median_y = np.median(y_acc)
        median_z = np.median(z_acc)
        energy_x = np.sum(x_acc**2)
        energy_y = np.sum(y_acc**2)
        energy_z = np.sum(z_acc**2)
        skew_x = skew(x_acc) if std_x > 0 else 0
        skew_y = skew(y_acc) if std_y > 0 else 0
        skew_z = skew(z_acc) if std_z > 0 else 0
        kurt_x = kurtosis(x_acc) if std_x > 0 else 0
        kurt_y = kurtosis(y_acc) if std_y > 0 else 0
        kurt_z = kurtosis(z_acc) if std_z > 0 else 0

        # Gyro
        mean_x_g = np.mean(x_gyro)
        mean_y_g = np.mean(y_gyro)
        mean_z_g = np.mean(z_gyro)
        std_x_g = np.std(x_gyro)
        std_y_g = np.std(y_gyro)
        std_z_g = np.std(z_gyro)
        max_x_g = np.max(x_gyro)
        max_y_g = np.max(y_gyro)
        max_z_g = np.max(z_gyro)
        min_x_g = np.min(x_gyro)
        min_y_g = np.min(y_gyro)
        min_z_g = np.min(z_gyro)
        range_x_g = max_x_g - min_x_g
        range_y_g = max_y_g - min_y_g
        range_z_g = max_z_g - min_z_g
        median_x_g = np.median(x_gyro)
        median_y_g = np.median(y_gyro)
        median_z_g = np.median(z_gyro)
        energy_x_g = np.sum(x_gyro**2)
        energy_y_g = np.sum(y_gyro**2)
        energy_z_g = np.sum(z_gyro**2)

        # Magnitud
        mag = np.sqrt(x_acc**2 + y_acc**2 + z_acc**2)
        mag_gyro = np.sqrt(x_gyro**2 + y_gyro**2 + z_gyro**2)
        mean_mag = np.mean(mag)
        mean_mag_gyro = np.mean(mag_gyro)
        std_mag = np.std(mag)
        std_mag_gyro = np.std(mag_gyro)
        median_mag = np.median(mag)
        median_mag_gyro = np.median(mag_gyro)
        skew_mag = skew(mag) if std_mag > 0 else 0
        kurt_mag = kurtosis(mag) if std_mag > 0 else 0

        # Derivadas
        mag_deriv = np.gradient(mag) * fs
        mean_mag_deriv = np.mean(mag_deriv) if len(mag_deriv) > 0 else 0
        std_mag_deriv = np.std(mag_deriv) if len(mag_deriv) > 0 else 0
        mag_gyro_deriv = np.diff(mag_gyro) * fs
        mean_mag_gyro_deriv = np.mean(mag_gyro_deriv) if len(mag_gyro_deriv) > 0 else 0
        std_mag_gyro_deriv = np.std(mag_gyro_deriv) if len(mag_gyro_deriv) > 0 else 0

        # Entropía
        hist_x, _ = np.histogram(x_acc, bins=10, density=True)
        hist_y, _ = np.histogram(y_acc, bins=10, density=True)
        hist_z, _ = np.histogram(z_acc, bins=10, density=True)
        entropy_x = entropy(hist_x + 1e-10)
        entropy_y = entropy(hist_y + 1e-10)
        entropy_z = entropy(hist_z + 1e-10)
        hist_x_g, _ = np.histogram(x_gyro, bins=10, density=True)
        hist_y_g, _ = np.histogram(y_gyro, bins=10, density=True)
        hist_z_g, _ = np.histogram(z_gyro, bins=10, density=True)
        entropy_x_g = entropy(hist_x_g + 1e-10)
        entropy_y_g = entropy(hist_y_g + 1e-10)
        entropy_z_g = entropy(hist_z_g + 1e-10)

        # Correlaciones
        corr_xy = np.corrcoef(x_acc, y_acc)[0, 1] if std_x > 0 and std_y > 0 else 0
        corr_xz = np.corrcoef(x_acc, z_acc)[0, 1] if std_x > 0 and std_z > 0 else 0
        corr_yz = np.corrcoef(y_acc, z_acc)[0, 1] if std_y > 0 and std_z > 0 else 0
        corr_xy_g = np.corrcoef(x_gyro, y_gyro)[0, 1] if std_x_g > 0 and std_y_g > 0 else 0
        corr_xz_g = np.corrcoef(x_gyro, z_gyro)[0, 1] if std_x_g > 0 and std_z_g > 0 else 0
        corr_yz_g = np.corrcoef(y_gyro, z_gyro)[0, 1] if std_y_g > 0 and std_z_g > 0 else 0

        # FFT y energías
        fft_x = np.fft.fft(x_acc)
        fft_y = np.fft.fft(y_acc)
        fft_z = np.fft.fft(z_acc)
        mag_fft_x = np.abs(fft_x)
        mag_fft_y = np.abs(fft_y)
        mag_fft_z = np.abs(fft_z)
        fft_x_g = np.fft.fft(x_gyro)
        fft_y_g = np.fft.fft(y_gyro)
        fft_z_g = np.fft.fft(z_gyro)
        mag_fft_x_g = np.abs(fft_x_g)
        mag_fft_y_g = np.abs(fft_y_g)
        mag_fft_z_g = np.abs(fft_z_g)

        mean_fft_x = np.mean(mag_fft_x)
        mean_fft_y = np.mean(mag_fft_y)
        mean_fft_z = np.mean(mag_fft_z)
        std_fft_x = np.std(mag_fft_x)
        std_fft_y = np.std(mag_fft_y)
        std_fft_z = np.std(mag_fft_z)
        mean_fft_x_g = np.mean(mag_fft_x_g)
        mean_fft_y_g = np.mean(mag_fft_y_g)
        mean_fft_z_g = np.mean(mag_fft_z_g)
        std_fft_x_g = np.std(mag_fft_x_g)
        std_fft_y_g = np.std(mag_fft_y_g)
        std_fft_z_g = np.std(mag_fft_z_g)

        energy_fft_x = np.sum(mag_fft_x**2)
        energy_fft_y = np.sum(mag_fft_y**2)
        energy_fft_z = np.sum(mag_fft_z**2)
        energy_fft_x_g = np.sum(mag_fft_x_g**2)
        energy_fft_y_g = np.sum(mag_fft_y_g**2)
        energy_fft_z_g = np.sum(mag_fft_z_g**2)

        hist_fft_x, _ = np.histogram(mag_fft_x, bins=10, density=True)
        hist_fft_y, _ = np.histogram(mag_fft_y, bins=10, density=True)
        hist_fft_z, _ = np.histogram(mag_fft_z, bins=10, density=True)
        entropy_fft_x = entropy(hist_fft_x + 1e-10)
        entropy_fft_y = entropy(hist_fft_y + 1e-10)
        entropy_fft_z = entropy(hist_fft_z + 1e-10)
        hist_fft_x_g, _ = np.histogram(mag_fft_x_g, bins=10, density=True)
        hist_fft_y_g, _ = np.histogram(mag_fft_y_g, bins=10, density=True)
        hist_fft_z_g, _ = np.histogram(mag_fft_z_g, bins=10, density=True)
        entropy_fft_x_g = entropy(hist_fft_x_g + 1e-10)
        entropy_fft_y_g = entropy(hist_fft_y_g + 1e-10)
        entropy_fft_z_g = entropy(hist_fft_z_g + 1e-10)

        num_coeffs = 5
        first_coeffs_x = mag_fft_x[:num_coeffs].tolist()
        first_coeffs_y = mag_fft_y[:num_coeffs].tolist()
        first_coeffs_z = mag_fft_z[:num_coeffs].tolist()


        norm_mean = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
        angle_x = np.arccos(mean_x / norm_mean) if norm_mean > 0 else 0
        angle_y = np.arccos(mean_y / norm_mean) if norm_mean > 0 else 0
        angle_z = np.arccos(mean_z / norm_mean) if norm_mean > 0 else 0
        norm_mean_g = np.sqrt(mean_x_g**2 + mean_y_g**2 + mean_z_g**2)
        angle_x_g = np.arccos(mean_x_g / norm_mean_g) if norm_mean_g > 0 else 0
        angle_y_g = np.arccos(mean_y_g / norm_mean_g) if norm_mean_g > 0 else 0
        angle_z_g = np.arccos(mean_z_g / norm_mean_g) if norm_mean_g > 0 else 0

        pitch = np.arctan2(x_acc, np.sqrt(y_acc**2 + z_acc**2))
        roll = np.arctan2(y_acc, z_acc)
        pitch_mean = np.mean(pitch)
        pitch_std = np.std(pitch)
        roll_mean = np.mean(roll)
        roll_std = np.std(roll)
        pitch_g = np.arctan2(x_gyro, np.sqrt(y_gyro**2 + z_gyro**2))
        roll_g = np.arctan2(y_gyro, z_gyro)
        pitch_mean_g = np.mean(pitch_g)
        pitch_std_g = np.std(pitch_g)
        roll_mean_g = np.mean(roll_g)
        roll_std_g = np.std(roll_g)

        iqr_x = np.percentile(x_acc, 75) - np.percentile(x_acc, 25)
        iqr_y = np.percentile(y_acc, 75) - np.percentile(y_acc, 25)
        iqr_z = np.percentile(z_acc, 75) - np.percentile(z_acc, 25)
        iqr_mag = np.percentile(mag, 75) - np.percentile(mag, 25)


        zcr_x = np.sum(np.diff(np.sign(x_acc)) != 0) / len(x_acc)
        zcr_y = np.sum(np.diff(np.sign(y_acc)) != 0) / len(y_acc)
        zcr_z = np.sum(np.diff(np.sign(z_acc)) != 0) / len(z_acc)

        peaks, _ = find_peaks(mag)
        num_peaks = len(peaks)

        def energy_bands(signal, fs=50):
            bands = [
                (0.0, 2.0),
                (2.0, 4.0),
                (4.0, 8.0),
                (8.0, 16.0),
                (16.0, 24.9)
            ]
            energies = []
            for low, high in bands:
                if low == 0.0:
                    b, a = butter(4, high / (fs/2), btype='low')
                else:
                    b, a = butter(4, [low / (fs/2), high / (fs/2)], btype='band')
                filtered = filtfilt(b, a, signal)
                energies.append(np.sum(filtered**2))
            total_energy = np.sum(energies) + 1e-10
            rel_energies = [e / total_energy for e in energies]
            return rel_energies

        w_x = energy_bands(x_acc, fs)
        w_y = energy_bands(y_acc, fs)
        w_z = energy_bands(z_acc, fs)
        w_mag = energy_bands(mag, fs)
        w_xg = energy_bands(x_gyro, fs)
        w_yg = energy_bands(y_gyro, fs)
        w_zg = energy_bands(z_gyro, fs)
        w_magg = energy_bands(mag_gyro, fs)

        features.append([
            mean_x, mean_y, mean_z,
            std_x, std_y, std_z,
            max_x, max_y, max_z,
            min_x, min_y, min_z,
            range_x, range_y, range_z,
            median_x, median_y, median_z,
            energy_x, energy_y, energy_z,
            skew_x, skew_y, skew_z,
            kurt_x, kurt_y, kurt_z,
            std_x_g, std_y_g, std_z_g,
            max_x_g, max_y_g, max_z_g,
            min_x_g, min_y_g, min_z_g,
            range_x_g, range_y_g, range_z_g,
            median_x_g, median_y_g, median_z_g,
            energy_x_g, energy_y_g, energy_z_g,
            entropy_x, entropy_y, entropy_z,
            entropy_x_g, entropy_y_g, entropy_z_g,
            corr_xy, corr_xz, corr_yz,
            corr_xy_g, corr_xz_g, corr_yz_g,
            mean_mag, std_mag, median_mag,
            mean_mag_gyro, std_mag_gyro, median_mag_gyro,
            skew_mag, kurt_mag,
            mean_mag_deriv, std_mag_deriv,
            mean_mag_gyro_deriv, std_mag_gyro_deriv,
            mean_fft_x, mean_fft_y, mean_fft_z,
            std_fft_x, std_fft_y, std_fft_z,
            energy_fft_x, energy_fft_y, energy_fft_z,
            std_fft_x_g, std_fft_y_g, std_fft_z_g,
            energy_fft_x_g, energy_fft_y_g, energy_fft_z_g,
            entropy_fft_x, entropy_fft_y, entropy_fft_z,
            entropy_fft_x_g, entropy_fft_y_g, entropy_fft_z_g,
            *first_coeffs_x, *first_coeffs_y, *first_coeffs_z,
            angle_x, angle_y, angle_z,
            angle_x_g, angle_y_g, angle_z_g,
            pitch_mean, pitch_std, roll_mean, roll_std,
            pitch_mean_g, pitch_std_g, roll_mean_g, roll_std_g,
            iqr_x, iqr_y, iqr_z, iqr_mag,
            zcr_x, zcr_y, zcr_z,
            num_peaks,
            *w_x, *w_y, *w_z, *w_mag,
            *w_xg, *w_yg, *w_zg, *w_magg
        ])

        labels_window.append(np.bincount(label_win).argmax() if len(label_win)>0 else 0)

    features_array = np.array(features)
    labels_array = np.array(labels_window)
    return features_array, labels_array


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------

def train_and_save_model(csv_path='data_train/har_train.csv', model_path='models/rf_har_model.joblib'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo de entrenamiento no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    # Intentar normalizar nombres: aceptar tanto acc_x como body_acc_x, etc.
    col_map = {}
    if 'acc_x' in df.columns:
        col_map.update({'acc_x':'body_acc_x','acc_y':'body_acc_y','acc_z':'body_acc_z'})
    if 'gyro_x' in df.columns:
        col_map.update({'gyro_x':'gyro_x','gyro_y':'gyro_y','gyro_z':'gyro_z'})
    df = df.rename(columns=col_map)

    # Asegurar que exista 'label'
    if 'label' not in df.columns:
        raise ValueError("El CSV debe contener columna 'label' con valores 1..6")

    X, y = calcular_caracteristicas_avanzadas_por_ventanas(df, fs=50, window_s=2.56, overlap=0.5)

    # dividir
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    print("Entrenando RandomForest... esto puede tardar unos minutos")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == '__main__':
    train_and_save_model()


