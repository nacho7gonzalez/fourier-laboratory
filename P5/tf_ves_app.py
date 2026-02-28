import dearpygui.dearpygui as dpg
from math import sin, cos
import pandas as pd
import time
import dearpygui_ext.themes as dpg_ext
import sys
from ble_interface import BLEInterface
from ble_interface import *
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch, butter, filtfilt 
import warnings

from config import *

EPS_STD = 1e-8

def safe_skew(x):
    std = np.std(x)
    if std < EPS_STD or np.allclose(x, x[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(skew(x))

def safe_kurtosis(x):
    std = np.std(x)
    if std < EPS_STD or np.allclose(x, x[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(kurtosis(x))

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

class TFVesGUI:
    
    def __init__(self):
        # Inicializar la interfaz BLE   
        self.ble_interface = BLEInterface()
        
        # Dataframes para datos ADC e IMU (para graficar)
        self.df_har = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        self.df_adc = pd.DataFrame(columns=['adc'])

        # Ventanas y buffers para graficar
        self.acc_window_size = 600  
        self.acc_indices = {label: 0 for label in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
        self.acc_last_update = time.time()
        self.acc_freq_hz = 50
        self.acc_dt = 1.0 / self.acc_freq_hz

        self.adc_index = 0
        self.adc_last_update = time.time()
        self.adc_freq_hz = 200
        self.adc_dt = 1.0 / self.adc_freq_hz
        self.adc_window_size = 5 * self.adc_freq_hz
        self._adc_window = [0.0] * self.adc_window_size

        # Buffer global de muestras para guardar: lista de dicts
        # cada fila: {'sample_idx', 't', 'adc', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'}
        self.data_buffer = []
        self.sample_counter = 0  # contador incremental de muestras globales

        # Mantener últimos valores conocidos para evitar campos vacíos
        # Si no hay valor anterior, se usa 0.0 (evita comas vacías).
        self.last_values = {
            'adc': 0.0,
            'acc_x': 0.0,
            'acc_y': 0.0,
            'acc_z': 0.0,
            'gyro_x': 0.0,
            'gyro_y': 0.0,
            'gyro_z': 0.0
        }

        # Valores por defecto para interfaz
        self.default_Rpot = 10000
        
        self.model_path = 'models/rf_har_model.joblib'
        self.clf = None
        self.scaler = None
        try:
            if os.path.exists(self.model_path):
                m = joblib.load(self.model_path)
                self.clf = m.get('model')
                self.scaler = m.get('scaler')
                print('Modelo HAR cargado.')
            else:
                print('Modelo HAR no encontrado, ejecuta train_classifier.py para crearlo.')
        except Exception as e:
            print('Error cargando modelo HAR:', e)

        # Estado local del filtro (None = desconocido, True/False = conocido)
        self.filter_enabled = None


    # ------------------------------------------------------------
    # Métodos para actualizar las gráficas con nuevos datos 
    # ------------------------------------------------------------
    def update_acc_plots_from_df_har_non_accum(self):
        if self.df_har.empty:
            return

        now = time.time()
        elapsed = now - self.acc_last_update
        samples_to_advance = int(np.ceil(elapsed * self.acc_freq_hz))
        if samples_to_advance <= 0:
            return

        self.acc_last_update += samples_to_advance * self.acc_dt
        imu_labels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        n_plot = min(samples_to_advance, len(self.df_har))

        for label in imu_labels: # Iterar sobre los 6 ejes
            if not hasattr(self, f'_acc_window_{label}'):
                # inicializar ventana (quitado el "[7]" erróneo)
                setattr(self, f'_acc_window_{label}', [0.0] * self.acc_window_size)

            y_window = getattr(self, f'_acc_window_{label}')
            new_vals = self.df_har[label].iloc[:n_plot].tolist()
            # Si new_vals tiene menos elementos que n_plot (por seguridad), rellenar con ceros
            if len(new_vals) < n_plot:
                new_vals = new_vals + [0.0] * (n_plot - len(new_vals))

            y_window = y_window[n_plot:] + new_vals
            setattr(self, f'_acc_window_{label}', y_window)
            
            x_vals = list(range(self.acc_window_size))

            # Uso de etiquetas dinámicas (f-strings) para la serie y el eje X
            try:
                dpg.set_value(f"acc_series_{label}", [x_vals, y_window])
                dpg.set_axis_limits(f"x_axis_{label}", x_vals[0], x_vals[-1])
            except Exception:
                # en caso de que la GUI no esté lista aún, ignorar
                pass

            self.acc_indices[label] = x_vals[-1] + 1

        self.df_har = self.df_har.iloc[n_plot:].reset_index(drop=True)
   
    
    def update_adc_plot_from_df_adc_non_accum(self):
        if self.df_adc.empty:
            return

        now = time.time()
        elapsed = now - self.adc_last_update
        samples_to_advance = int(np.ceil(elapsed * self.adc_freq_hz))
        if samples_to_advance <= 0:
            return

        self.adc_last_update += samples_to_advance * self.adc_dt
        n_plot = min(samples_to_advance, len(self.df_adc))

        new_vals = self.df_adc['adc'].iloc[:n_plot].tolist()
        if len(new_vals) < n_plot:
            new_vals = new_vals + [0.0] * (n_plot - len(new_vals))

        self._adc_window = self._adc_window[n_plot:] + new_vals
        x_vals = list(range(self.adc_window_size))
        try:
            dpg.set_value("adc_series", [x_vals, self._adc_window])
            dpg.set_axis_limits("x_axis_adc", x_vals[0], x_vals[-1])
        except Exception:
            pass
        
        self.adc_index = x_vals[-1] + 1

        self.df_adc = self.df_adc.iloc[n_plot:].reset_index(drop=True)


    # ------------------------------------------------------------
    # Conexión BLE
    # ------------------------------------------------------------
    def update_connected_status(self, connected):
        if connected:
            if not self.ble_interface.get_connection_status():
                connected = self.ble_interface.connect_by_name(DEVICE_NAME)
                if not connected:
                    print("No se pudo conectar al dispositivo.")
                    dpg.set_value("status_bar", "No se pudo conectar al dispositivo.")
                    return
                else:
                    dpg.set_value("status_bar", "Conectado. Leyendo estado del filtro...")
                    # leer estado del filtro luego de conectar
                    self.read_filter_from_device()
        else:
            if self.ble_interface.get_connection_status():
                self.ble_interface.disconnect()
                connected = False
                dpg.set_value("status_bar", "Desconectado.")
        dpg.set_value("Connected", connected)
        

    # ------------------------------------------------------------
    # Update gain (slider) - mantiene cálculo G y BLE write
    # ------------------------------------------------------------
    def update_gain(self, sender, app_data, user_data=None):
        try:
            R_pot = float(app_data)
        except Exception:
            R_pot = 0.0

        denom = (100.0 + R_pot + 330.0)
        if denom == 0.0:
            R_G = float('inf')
        else:
            R_G = (100.0 + R_pot) * 330.0 / denom

        if R_G == 0.0 or R_G == float('inf'):
            G = float('inf')
        else:
            G = 1.0 + (100000.0 / R_G)

        try:
            dpg.set_value("G_display", float(G))
        except Exception:
            pass

        gain_to_write = int(R_pot)
        if self.ble_interface.get_connection_status():
            try:
                self.ble_interface.write_characteristic(GAIN_CHARACTERISTIC_UUID, [gain_to_write], 'int16')
                gain_value = self.ble_interface.read_characteristic(GAIN_CHARACTERISTIC_UUID, 'int16')
                print('Gain set to (from BLE):', gain_value)
            except Exception as e:
                print("Error escribiendo/leyendo ganancia via BLE:", e)
        else:
            # Se omite spam en consola si no interesa; mantener para debug
            print("Dispositivo no conectado. No se puede setear la ganacia via BLE.")  


    # ------------------------------------------------------------
    # Subscripciones BLE
    # ------------------------------------------------------------
    def update_imu_subscription(self, subscribed):
        if self.ble_interface.get_connection_status():
            char_uuid = IMU_CHARACTERISTIC_UUID
            handler = self.imu_notification_handler
            tag = "Subscribed_IMU"
            columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            self.df_har = pd.DataFrame(columns=columns)

            if subscribed:
                self.ble_interface.subscribe_to_char_notifications(
                    char_uuid, handler, data_type='int16'
                )
                dpg.set_value(tag, True)
                self.acc_indices = {label: 0 for label in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
                self.acc_last_update = time.time()
                print("Subscribed to IMU notifications.")
            else:
                self.ble_interface.unsubscribe_to_char_notifications(char_uuid)
                dpg.set_value(tag, False)
                print("Unsubscribed from IMU notifications.")
        else:
            print("Device not connected. Cannot subscribe to IMU.")

    def update_adc_subscription(self, subscribed):
        if self.ble_interface.get_connection_status():
            char_uuid = ADC_CHARACTERISTIC_UUID
            handler = self.adc_notification_handler
            tag = "Subscribed_ADC"
            columns = ['adc']
            self.df_adc = pd.DataFrame(columns=columns)

            if subscribed:
                self.ble_interface.subscribe_to_char_notifications(
                    char_uuid, handler, data_type='int16'
                )
                dpg.set_value(tag, True)
                print("Subscribed to ADC notifications.")
            else:
                self.ble_interface.unsubscribe_to_char_notifications(char_uuid)
                dpg.set_value(tag, False)
                self.adc_index = 0
                self.adc_last_update = time.time()
                print("Unsubscribed from ADC notifications.")
        else:
            print("Device not connected. Cannot subscribe to ADC.")


    # ------------------------------------------------------------
    # Callbacks BLE: al recibir datos, ahora añado filas completas usando last_values
    # ------------------------------------------------------------
    def adc_notification_handler(self, sender, data):
        samples = data
        # convertir a voltios
        new_data = pd.DataFrame({'adc': samples})
        try:
            new_data = (new_data / ADC_RESOLUTION_STEPS) * ADC_REFERENCE_VOLTAGE
        except Exception:
            # si las constantes no están definidas correctamente, dejar como están
            pass

        # append to df para graficar
        if self.df_adc.empty:
            self.df_adc = new_data
        else:
            self.df_adc = pd.concat([self.df_adc, new_data], ignore_index=True)

        # Agregar cada muestra al buffer global con sample_idx y timestamp
        # Para evitar campos vacíos, actualizamos last_values['adc'] primero y luego añadimos una fila "completa"
        for val in new_data['adc'].tolist():
            now = time.time()
            # actualizar último ADC conocido
            self.last_values['adc'] = float(val)

            row = {
                'sample_idx': int(self.sample_counter),
                't': now,
                'adc': float(self.last_values['adc']),
                'acc_x': float(self.last_values['acc_x']),
                'acc_y': float(self.last_values['acc_y']),
                'acc_z': float(self.last_values['acc_z']),
                'gyro_x': float(self.last_values['gyro_x']),
                'gyro_y': float(self.last_values['gyro_y']),
                'gyro_z': float(self.last_values['gyro_z'])
            }
            self.data_buffer.append(row)
            self.sample_counter += 1
        #print("ADC",self.df_adc.shape)


    def imu_notification_handler(self, sender, data):
        samples = data
        # Ahora esperamos 6 muestras interfoliadas (Acc X, Y, Z, Gyro X, Y, Z)
        new_data = pd.DataFrame({
            'acc_x': samples[0::6],
            'acc_y': samples[1::6],
            'acc_z': samples[2::6],
            'gyro_x': samples[3::6],
            'gyro_y': samples[4::6],
            'gyro_z': samples[5::6]
        })
        # convertir a g (o la escala definida en config)
        try:
            new_data = new_data / np.double(INT16_MAX_VALUE+1) * IMU_IMU_ACCEL_RANGE
        except Exception:
            pass
        if self.df_har.empty:
            self.df_har = new_data
        else:
            self.df_har = pd.concat([self.df_har, new_data], ignore_index=True)

        # Para cada muestra: actualizamos last_values y guardamos una fila completa usando último ADC conocido
        for ax, ay, az, gx, gy, gz in zip(
            new_data['acc_x'].tolist(), new_data['acc_y'].tolist(), new_data['acc_z'].tolist(),
            new_data['gyro_x'].tolist(), new_data['gyro_y'].tolist(), new_data['gyro_z'].tolist()
        ):
            now = time.time()
            self.last_values['acc_x'] = float(ax)
            self.last_values['acc_y'] = float(ay)
            self.last_values['acc_z'] = float(az)
            self.last_values['gyro_x'] = float(gx)
            self.last_values['gyro_y'] = float(gy)
            self.last_values['gyro_z'] = float(gz)

            row = {
                'sample_idx': int(self.sample_counter),
                't': now,
                'adc': float(self.last_values.get('adc', 0.0)),
                'acc_x': float(self.last_values['acc_x']),
                'acc_y': float(self.last_values['acc_y']),
                'acc_z': float(self.last_values['acc_z']),
                'gyro_x': float(self.last_values['gyro_x']),
                'gyro_y': float(self.last_values['gyro_y']),
                'gyro_z': float(self.last_values['gyro_z'])
            }
            self.data_buffer.append(row)
            self.sample_counter += 1
        #print("ACC",self.df_adc.shape)

    # ------------------------------------------------------------
    # Funcionalidades de guardado solicitadas
    #  - guardar últimos N samples (500)
    #  - guardar por índices de sample (start,end)
    #  Salida: columnas ordenadas ACC_X, ACC_Y, ACC_Z, ADC, IDX
    # ------------------------------------------------------------
    def save_last_n_samples(self, n=500):
        if not self.data_buffer:
            print("Buffer vacío: no hay datos para guardar.")
            return

        n = int(n)
        df = pd.DataFrame(self.data_buffer)
        df_sorted = df.sort_values('sample_idx').reset_index(drop=True)
        df_sel = df_sorted.tail(n).copy()

        # Preparar columnas y orden requerido: ACC_X, ACC_Y, ACC_Z, ADC, IDX
        out = pd.DataFrame({
            'ACC_X': df_sel['acc_x'].astype(float),
            'ACC_Y': df_sel['acc_y'].astype(float),
            'ACC_Z': df_sel['acc_z'].astype(float),
            'GYRO_X': df_sel['gyro_x'].astype(float),
            'GYRO_Y': df_sel['gyro_y'].astype(float),
            'GYRO_Z': df_sel['gyro_z'].astype(float),
            'ADC': df_sel['adc'].astype(float),
            'IDX': df_sel['sample_idx'].astype(int)
        })

        
        filename_base = dpg.get_value("record_filename") or "datos_medidos"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"{filename_base}_last{n}_{timestamp_str}.csv"
        
        os.makedirs("datos", exist_ok=True)
        fullpath = os.path.join("datos", filename)
        
        try:
            out.to_csv(fullpath, index=False)
            dpg.set_value("record_last_file", fullpath)
            print(f"Últimos {n} samples guardados en {fullpath} (filas: {len(out)})")
        except Exception as e:
            print("Error al guardar CSV:", e)

    def save_by_index_range(self, start_idx, end_idx):
        if not self.data_buffer:
            print("Buffer vacío: no hay datos para guardar.")
            return

        try:
            start_idx = int(start_idx)
        except Exception:
            print("Start index inválido.")
            return
        try:
            end_idx = int(end_idx)
        except Exception:
            print("End index inválido.")
            return

        if end_idx < start_idx:
            print("End index debe ser >= start index.")
            return

        df = pd.DataFrame(self.data_buffer)
        df_sorted = df.sort_values('sample_idx').reset_index(drop=True)

        df_sel = df_sorted[(df_sorted['sample_idx'] >= start_idx) & (df_sorted['sample_idx'] <= end_idx)].copy()

        if df_sel.empty:
            print("No se encontraron muestras en el rango solicitado.")
            return

        # Preparar columnas y orden requerido: ACC_X, ACC_Y, ACC_Z, ADC, IDX
        out = pd.DataFrame({
            'ACC_X': df_sel['acc_x'].astype(float),
            'ACC_Y': df_sel['acc_y'].astype(float),
            'ACC_Z': df_sel['acc_z'].astype(float),
            'GYRO_X': df_sel['gyro_x'].astype(float),
            'GYRO_Y': df_sel['gyro_y'].astype(float),
            'GYRO_Z': df_sel['gyro_z'].astype(float),
            'ADC': df_sel['adc'].astype(float),
            'IDX': df_sel['sample_idx'].astype(int)
        })


        filename_base = dpg.get_value("record_filename") or "datos_medidos"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"{filename_base}_idx{start_idx}-{end_idx}_{timestamp_str}.csv"
        
        os.makedirs("datos", exist_ok=True)
        fullpath = os.path.join("datos", filename)
        try:
            out.to_csv(fullpath, index=False)
            dpg.set_value("record_last_file", fullpath)
            print(f"Muestras [{start_idx} .. {end_idx}] guardadas en {fullpath} (filas: {len(out)})")
        except Exception as e:
            print("Error al guardar CSV:", e)
            
    def _build_df_for_feature(self, df_window):
    # df_window es un DataFrame con columnas acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
        tmp = pd.DataFrame()
        tmp['body_acc_x'] = df_window['acc_x'].values
        tmp['body_acc_y'] = df_window['acc_y'].values
        tmp['body_acc_z'] = df_window['acc_z'].values
        tmp['gyro_x'] = df_window['gyro_x'].values
        tmp['gyro_y'] = df_window['gyro_y'].values
        tmp['gyro_z'] = df_window['gyro_z'].values
        tmp['label'] = 0 # dummy
        return tmp
    def classify_recent_window(self, fs=50, window_s=4):
        """
        Clasifica la actividad usando las últimas window_s segundos.
        Muestra resultado en tag 'predicted_activity' (asegúrate de crear ese tag en la GUI).
        """
        if self.clf is None or self.scaler is None:
            dpg.set_value('predicted_activity', 'Modelo no cargado')
            print("Clasificar: modelo no cargado.")
            return

        need_samples = int(fs * window_s)
        if len(self.data_buffer) < need_samples:
            dpg.set_value('predicted_activity', 'Insuficientes muestras')
            print("Clasificar: insuficientes muestras:", len(self.data_buffer), "necesito", need_samples)
            return

        # construir ventana con las últimas muestras
        df_buf = pd.DataFrame(self.data_buffer[-need_samples:])
        df_win = df_buf[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].reset_index(drop=True)

        # construir df compatible con la extractor
        tmp = pd.DataFrame({
            'body_acc_x': df_win['acc_x'].values,
            'body_acc_y': df_win['acc_y'].values,
            'body_acc_z': df_win['acc_z'].values,
            'gyro_x': df_win['gyro_x'].values,
            'gyro_y': df_win['gyro_y'].values,
            'gyro_z': df_win['gyro_z'].values,
            'label': np.zeros(len(df_win), dtype=int)
        })

        # extraer features SIN overlap (overlap=0.0) para obtener exactamente 1 ventana si corresponde
        X_feat, _ = calcular_caracteristicas_avanzadas_por_ventanas(tmp, fs=fs, window_s=window_s, overlap=0.0)
        if X_feat.size == 0:
            dpg.set_value('predicted_activity', 'No features')
            print("Clasificar: X_feat vacío")
            return

        # debug: comprobar shapes y rangos
        try:
            print("DEBUG: X_feat.shape =", X_feat.shape)
            print("DEBUG: X_feat min/mean/max (first 8 cols):", np.min(X_feat,axis=0)[:8], np.mean(X_feat,axis=0)[:8], np.max(X_feat,axis=0)[:8])
        except Exception:
            pass

        # verificar correspondencia con modelo
        n_model = getattr(self.clf, 'n_features_in_', None)
        if n_model is not None and X_feat.shape[1] != n_model:
            msg = f"Feature mismatch: extractor produce {X_feat.shape[1]} cols, modelo espera {n_model}"
            dpg.set_value('predicted_activity', msg)
            print(msg)
            return

        # escalar y predecir
        try:
            Xs = self.scaler.transform(X_feat)
        except Exception as e:
            print("Error escalando features:", e)
            dpg.set_value('predicted_activity', 'Error escalando features')
            return

        try:
            y_pred = self.clf.predict(Xs)
            proba = self.clf.predict_proba(Xs) if hasattr(self.clf, 'predict_proba') else None
        except Exception as e:
            print("Error predict:", e)
            dpg.set_value('predicted_activity', 'Error predict')
            return

        pred = int(np.bincount(y_pred).argmax())  # moda si más de una ventana
        mapping = {1:'CAMINANDO',2:'SUBIENDO ESCALERAS',3:'BAJANDO ESCALERAS',4:'SENTADO',5:'DE PIE',6:'ACOSTADO'}
        text = f"{pred} - {mapping.get(pred,'?')}"
        if proba is not None:
            text += f' (p={proba[0].max():.2f})'
        dpg.set_value('predicted_activity', text)
        print("Clasificación:", text)



    # ------------------------------------------------------------
    # Activar filtro de firmware por BLE
    # ------------------------------------------------------------
    def update_filter_status_display(self):
        """Actualiza el widget que muestra si el filtro está ON/OFF."""
        try:
            if self.filter_enabled is None:
                dpg.set_value("filter_status", "DESCONOCIDO ")
            elif self.filter_enabled:
                dpg.set_value("filter_status", "ON")
            else:
                dpg.set_value("filter_status", "OFF")
        except Exception:
            pass

    def read_filter_from_device(self):
        """Intenta leer la característica filtro desde la placa (1 byte: 0/1).
        Maneja distintos formatos de retorno: int, list/tuple, bytes/bytearray, str.
        """
        if not self.ble_interface.get_connection_status():
            self.filter_enabled = None
            dpg.set_value("status_bar", "No conectado: no se puede leer filtro")
            self.update_filter_status_display()
            return

        try:
            # Intentar leer sin codec (la implementación de BLEInterface puede aceptar o no el codec)
            try:
                val = self.ble_interface.read_characteristic(FILTRO_CHARACTERISTIC_UUID)
            except TypeError:
                # Si la interfaz exige un segundo argumento, probar 'uint8' y si falla se captura abajo
                try:
                    val = self.ble_interface.read_characteristic(FILTRO_CHARACTERISTIC_UUID, 'uint8')
                except Exception:
                    # último recurso: leer sin codec
                    val = self.ble_interface.read_characteristic(FILTRO_CHARACTERISTIC_UUID)

            # Normalizar el valor leído a un int
            v = None
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    raise ValueError("read returned empty list/tuple")
                v = int(val[0])
            elif isinstance(val, (bytes, bytearray)):
                if len(val) == 0:
                    raise ValueError("read returned empty bytes")
                # interpretar como entero little-endian
                v = int(val[0])
            elif isinstance(val, str):
                # intentar convertir string que contenga dígitos
                v = int(val.strip())
            elif isinstance(val, int):
                v = int(val)
            else:
                # último intento: tratar de convertir directamente
                v = int(val)

            self.filter_enabled = (v != 0)
            dpg.set_value("status_bar", f"Filtro leído: {'ON' if self.filter_enabled else 'OFF'}")
        except Exception as e:
            print("Error leyendo filtro:", e)
            dpg.set_value("status_bar", f"Error leyendo filtro: {e}")
            self.filter_enabled = None
        finally:
            self.update_filter_status_display()

    def toggle_filter_callback(self, sender, app_data, user_data):
        """Cambia el estado del filtro: escribe al dispositivo y actualiza GUI.
        Usa bytes() para la escritura (evita codec 'uint8' no implementado).
        """
        if not self.ble_interface.get_connection_status():
            dpg.set_value("status_bar", "No conectado: no se puede cambiar filtro")
            return

        # Si desconocido, leer primero
        if self.filter_enabled is None:
            self.read_filter_from_device()
            if self.filter_enabled is None:
                return

        new_state = not bool(self.filter_enabled)
        byte_to_write = 1 if new_state else 0

        wrote_ok = False
        last_exc = None

        # Intento 1: escribir como bytes (recomendado)
        try:
            self.ble_interface.write_characteristic(FILTRO_CHARACTERISTIC_UUID, bytes([byte_to_write]))
            wrote_ok = True
        except Exception as e:
            last_exc = e

        # Intento 2: escribir como lista de enteros (fallback)
        if not wrote_ok:
            try:
                self.ble_interface.write_characteristic(FILTRO_CHARACTERISTIC_UUID, [byte_to_write])
                wrote_ok = True
            except Exception as e:
                last_exc = e

        if not wrote_ok:
            print("Error escribiendo filtro via BLE (todos los intentos fallaron):", last_exc)
            dpg.set_value("status_bar", f"Error setear filtro: {last_exc}")
            return

        # Pequeña pausa para que el dispositivo procese
        time.sleep(0.05)

        # Leer de vuelta para confirmar (usar la función robusta)
        try:
            self.read_filter_from_device()
            dpg.set_value("status_bar", f"Filtro seteado: {'ON' if self.filter_enabled else 'OFF'}")
        except Exception as e:
            print("Error leyendo filtro despues de escribir:", e)
            dpg.set_value("status_bar", f"Filtro escrito, lectura fallo: {e}")



    # ------------------------------------------------------------
    # Crear la GUI con DearPyGui (incluye controles de guardado por request)
    # ------------------------------------------------------------
    def create_gui(self):
        with dpg.window(label="TF_VES", tag="win", autosize=True) as primary_window:

            with dpg.group(horizontal=True):
                # Controles a la izquierda
                with dpg.group(horizontal=False, label="Controls", width=340):
                    dpg.add_checkbox(label="Connected", tag='Connected', enabled=True, default_value=False)
                    dpg.add_button(label='Connect',  callback=lambda: (print("Connect button pressed"), self.update_connected_status(True)))
                    dpg.add_button(label='Disconnect', callback=lambda: (print("Disconnect button pressed"), self.update_connected_status(False)))
                    
                    dpg.add_spacer(height=10)
                    dpg.add_text("Ajuste de Ganancia")

                    dpg.add_slider_int(
                        label="Resistencia Potenciómetro",
                        default_value=self.default_Rpot,
                        min_value=390,
                        max_value=32767,
                        callback=self.update_gain,
                        clamped=True,
                        width=300,
                        tag="R_pot_slider")
                    
                    # G display
                    R_pot_init = float(self.default_Rpot)
                    denom = (100.0 + R_pot_init + 330.0)
                    R_G_init = (100.0 + R_pot_init) * 330.0 / denom if denom != 0.0 else float('inf')
                    G_init = 1.0 + (100000.0 / R_G_init) if (R_G_init != 0.0 and R_G_init != float('inf')) else float('inf')
                    dpg.add_input_float(
                        label="Ganancia G",
                        tag="G_display",
                        default_value=float(G_init),
                        readonly=True,
                        format="%.4f"
                    )

                    dpg.add_separator()
                    dpg.add_text("Guardado de datos (CSV)")
                    dpg.add_input_text(label="Nombre base archivo", tag="record_filename", default_value="datos_medidos", width=260)

                    # Guardar últimos N samples (botón)
                    dpg.add_spacing(count=1)
                    dpg.add_input_int(label="N samples a guardar (últimos)", tag="save_last_n", default_value=500, min_value=1, width=200)
                    dpg.add_button(label="Guardar últimos N samples", callback=lambda s, a, u: self.save_last_n_samples(dpg.get_value("save_last_n")))

                    dpg.add_separator()
                    dpg.add_text("Guardar por índices de sample (inclusive)")
                    dpg.add_input_int(label="Start index (inclusive)", tag="save_idx_start", default_value=0, min_value=0, width=200)
                    dpg.add_input_int(label="End index (inclusive)", tag="save_idx_end", default_value=1000, min_value=0, width=200)
                    dpg.add_button(label="Guardar por índices (start/end)", callback=lambda s, a, u: self.save_by_index_range(dpg.get_value("save_idx_start"), dpg.get_value("save_idx_end")))

                    dpg.add_separator()
                    dpg.add_text("Último archivo guardado:")
                    dpg.add_text("", tag="record_last_file")
                    
                    dpg.add_separator()
                    # Suscripciones IMU/ADC
                    dpg.add_checkbox(label="Subscribed (IMU)", tag='Subscribed_IMU', enabled=True, default_value=False)
                    dpg.add_button(label='Subscribe IMU', callback=lambda: self.update_imu_subscription(True))
                    dpg.add_button(label='Unsubscribe IMU', callback=lambda: self.update_imu_subscription(False))
                    dpg.add_spacer(height=6)
                    dpg.add_checkbox(label="Subscribed (ADC)", tag='Subscribed_ADC', enabled=True, default_value=False)
                    dpg.add_button(label='Subscribe ADC', callback=lambda: self.update_adc_subscription(True))
                    dpg.add_button(label='Unsubscribe ADC', callback=lambda: self.update_adc_subscription(False))
                    
                    dpg.add_text('Predicción de actividad:')
                    dpg.add_text('', tag='predicted_activity')
                    dpg.add_button(label='Clasificar ahora', callback=lambda: self.classify_recent_window())
                    dpg.add_checkbox(label='Auto clasificar (2.56s)', tag='auto_classify', default_value=False)

                    dpg.add_separator()
                    # --- NUEVOS controles para filtro ---
                    dpg.add_text("Filtro de G")
                    dpg.add_button(label="ON/OFF", callback=self.toggle_filter_callback)
                    dpg.add_input_text(label="Estado del filtro:", tag="filter_status", default_value="DESCONOCIDO", readonly=True, width=180)
                    dpg.add_spacer(height=6)
                    # FIN nuevos controles

                # Gráficas y barra de estado a la derecha
                with dpg.group(horizontal=False):
                    # Iterar sobre ACCELEROMETER y GYROSCOPE
                    imu_axes_list = [
                        ('acc_x', 'ACC X (g)'), ('acc_y', 'ACC Y (g)'), ('acc_z', 'ACC Z (g)'),
                        ('gyro_x', 'GYRO X (dps)'), ('gyro_y', 'GYRO Y (dps)'), ('gyro_z', 'GYRO Z (dps)') # Nuevos
                    ]
                    
                    for axis, label in imu_axes_list:
                        with dpg.plot(label=f"IMU {axis.upper()}", height=150, width=1100, tag=f"acc_plot_{axis}"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag=f"x_axis_{axis}")
                            dpg.set_axis_limits(dpg.last_item(), 0, self.acc_window_size)
                            with dpg.plot_axis(dpg.mvYAxis, label=label, tag=f"y_axis_{axis}") as y_axis:
                                dpg.add_line_series(
                                    list(range(self.acc_window_size)),
                                    list(np.zeros((self.acc_window_size))),
                                    label=axis.upper(),
                                    tag=f"acc_series_{axis}"
                                )
                                dpg.set_axis_limits_auto(y_axis)

                    with dpg.group(horizontal=False):
                        with dpg.plot(label="ADC", height=200, width=1100, tag="adc_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag="x_axis_adc")
                            dpg.set_axis_limits(dpg.last_item(), 0, self.adc_window_size)
                            with dpg.plot_axis(dpg.mvYAxis, label="ADC (V)") as y_axis:
                                dpg.add_line_series(
                                    list(range(self.adc_window_size)),
                                    [0.0] * self.adc_window_size,
                                    label="ADC",
                                    tag="adc_series"
                                ) 
                                dpg.set_axis_limits(y_axis, -0.5, 3.5)
                                 
                    with dpg.group(horizontal=False,label='Estado:'):
                        dpg.add_separator()
                        dpg.add_input_text(
                            label="Status",
                            tag="status_bar",
                            default_value="Bienvenido",
                            readonly=True,
                            multiline=True,
                            height=0,
                            width=-1
                        )
                        dpg.add_spacer(height=10)  
            

    def run(self):
        dpg.create_context()
        light_theme = dpg_ext.create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        dpg.create_viewport(title='Taller Fourier Vestible', width=1400, height=850)
        self.create_gui()
        dpg.set_primary_window("win", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._last_classify = time.time()
        # main loop
        while dpg.is_dearpygui_running():
            self.update_adc_plot_from_df_adc_non_accum()
            self.update_acc_plots_from_df_har_non_accum()
            dpg.render_dearpygui_frame()
            if dpg.get_value('auto_classify'):
                if time.time() - self._last_classify > 2.56:
                    try:
                        self.classify_recent_window()
                    except Exception as e:
                        print('Error clasificador:', e)
                    self._last_classify = time.time()

        # cleanup
        if self.ble_interface.get_connection_status():
            self.ble_interface.disconnect()
        dpg.destroy_context()
        



# Clase para redirigir stdout a barra de estado si se quisiera usar
class StatusBarWriter:
    def __init__(self, status_bar_tag):
        self.status_bar_tag = status_bar_tag
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                if line.strip():
                    dpg.set_value(self.status_bar_tag, line)
            self.buffer = lines[-1]

    def flush(self):
        if self.buffer.strip():
            dpg.set_value(self.status_bar_tag, self.buffer)
            self.buffer = ""


if __name__ == "__main__":
    app = TFVesGUI()
    app.run()
