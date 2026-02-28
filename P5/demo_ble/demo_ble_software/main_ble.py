from ble_interface import BLEInterface
from time import sleep
from queue import Queue
import threading

try:
    import dearpygui.dearpygui as dpg
except Exception as e:
    raise ImportError("DearPyGui no está instalado. Instala con: pip install dearpygui")

# UUIDs y nombre (ajusta si es necesario)
ADC_CHARACTERISTIC_UUID = "55c40000-0130-4dd1-be0c-40588193b485"
GAIN_CHARACTERISTIC_UUID = "55c40000-0330-4dd1-be0c-40588193b485"

# Nombre por defecto del dispositivo
XIAO_NAME = 'GRUPO 1 - XIAO nRF52840 Sense'

# Cola para pasar notificaciones desde el hilo BLE al hilo de la GUI
adc_queue = Queue()

# Handler de notificaciones (se ejecuta en el hilo del BLE)
def adc_notification_handler(sender, data):
    """Callback que añade las muestras recibidas a una cola para ser procesadas por la GUI."""
    try:
        # Guardamos una versión corta (primeras muestras) para mostrar en la GUI
        samples = list(data)
        preview = samples[:10]
        adc_queue.put((sender, preview))
    except Exception as e:
        adc_queue.put(("error", str(e)))


class BLEGuiApp:
    def __init__(self):
        self.ble = BLEInterface()
        self.device_name = XIAO_NAME

        dpg.create_context()
        self._build_ui()
        dpg.create_viewport(title='Interfaz', width=700, height=450)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Lanzar un hilo que compruebe la cola periódicamente para actualizar la lista en la GUI
        self.updater_thread = threading.Thread(target=self._queue_updater, daemon=True)
        self.updater_thread.start()

        dpg.start_dearpygui()
        dpg.destroy_context()

    def _build_ui(self):
        with dpg.window(label="Control BLE", width=680, height=420):
            dpg.add_text("Dispositivo:")
            dpg.add_input_text(default_value=self.device_name, tag="device_name_input")
            dpg.add_spacing(count=1)

            dpg.add_button(label="Conectar", callback=self.connect_cb)
            dpg.add_same_line()
            dpg.add_button(label="Desconectar", callback=self.disconnect_cb)
            dpg.add_same_line()
            dpg.add_button(label="Estado conexión", callback=self.status_cb)

            dpg.add_separator()

            dpg.add_text("Control de ganancia:")
            dpg.add_button(label="Leer ganancia", callback=self.read_gain_cb)
            dpg.add_same_line()
            dpg.add_input_int(label="Nueva ganancia", tag="gain_input", default_value=452)
            dpg.add_same_line()
            dpg.add_button(label="Escribir ganancia", callback=self.write_gain_cb)

            dpg.add_separator()

            dpg.add_text("Notificaciones ADC:")
            dpg.add_button(label="Suscribirse a ADC (notificaciones)", callback=self.subscribe_adc_cb)
            dpg.add_same_line()
            dpg.add_button(label="Cancelar suscripción ADC", callback=self.unsubscribe_adc_cb)
            dpg.add_same_line()
            dpg.add_button(label="Actualizar log ADC", callback=self.refresh_adc_log_cb)

            dpg.add_separator()

            dpg.add_text("Registro / Salida:", tag="log_label")
            dpg.add_input_text(multiline=True, readonly=True, height=200, tag="log_box")

    # ----------------- Callbacks GUI -----------------
    def connect_cb(self, sender, app_data):
        name = dpg.get_value("device_name_input")
        self._log(f"Conectando a {name}...")
        try:
            # Conectar en hilo para no bloquear la GUI
            threading.Thread(target=self._connect_thread, args=(name,), daemon=True).start()
        except Exception as e:
            self._log(f"Error al iniciar conexión: {e}")

    def _connect_thread(self, name):
        try:
            self.ble.connect_by_name(name)
            status = self.ble.get_connection_status()
            self._log(f"Conexión establecida: {status}")
        except Exception as e:
            self._log(f"Error conectando: {e}")

    def disconnect_cb(self, sender, app_data):
        try:
            self.ble.disconnect()
            self._log("Desconectado")
        except Exception as e:
            self._log(f"Error al desconectar: {e}")

    def status_cb(self, sender, app_data):
        try:
            status = self.ble.get_connection_status()
            self._log(f"Estado conexión: {status}")
        except Exception as e:
            self._log(f"Error obteniendo estado: {e}")

    def read_gain_cb(self, sender, app_data):
        try:
            gain = self.ble.read_characteristic(GAIN_CHARACTERISTIC_UUID, 'int16')
            self._log(f"Ganancia leída: {gain}")
        except Exception as e:
            self._log(f"Error leyendo ganancia: {e}")

    def write_gain_cb(self, sender, app_data):
        try:
            gain = dpg.get_value("gain_input")
            self.ble.write_characteristic(GAIN_CHARACTERISTIC_UUID, [int(gain)], 'int16')
            # Leer para verificar
            new_gain = self.ble.read_characteristic(GAIN_CHARACTERISTIC_UUID, 'int16')
            self._log(f"Ganancia escrita: {gain}. Verificación: {new_gain}")
        except Exception as e:
            self._log(f"Error escribiendo ganancia: {e}")

    def subscribe_adc_cb(self, sender, app_data):
        try:
            self.ble.subscribe_to_char_notifications(ADC_CHARACTERISTIC_UUID, adc_notification_handler, 'int16')
            self._log("Suscrito a ADC (notificaciones). Las muestras se añaden al log cuando pulse 'Actualizar log ADC'.")
        except Exception as e:
            self._log(f"Error suscribiendo: {e}")

    def unsubscribe_adc_cb(self, sender, app_data):
        try:
            # Intentar cancelar la suscripción escribiendo False si la interfaz lo soporta
            # Si BLEInterface tiene un método específico para unsubscirbe, reemplazarlo.
            self.ble.unsubscribe_char(ADC_CHARACTERISTIC_UUID)
            self._log("Cancelada suscripción a ADC (si es soportado)")
        except Exception as e:
            self._log(f"Error cancelando suscripción (quizá no implementado): {e}")

    def refresh_adc_log_cb(self, sender=None, app_data=None):
        # Vaciar la cola y añadir los elementos a la caja de log
        updated = False
        while not adc_queue.empty():
            sender_id, preview = adc_queue.get()
            self._log(f"Notificación desde {sender_id}: {preview}")
            updated = True
        if not updated:
            self._log("No hay nuevas notificaciones ADC en la cola.")

    # Hilo que revisa la cola y escribe entradas en el log periódicamente
    def _queue_updater(self):
        # No usamos timers de DPG, simplemente comprobamos la cola cada 0.5 s
        while True:
            if not adc_queue.empty():
                # No llamar a funciones de DPG desde este hilo; en su lugar, ponemos en la GUI
                # usando dpg.run_async o actualizamos el input_text directamente (es tolerable en muchos casos)
                # Para máxima compatibilidad, simplemente marcaremos que hay nuevos datos y el usuario puede "Actualizar log ADC"
                # Sin bloqueo: dormimos poco para no consumir CPU
                pass
            sleep(0.5)

    def _log(self, text):
        # Añade una entrada al final del textbox de log
        try:
            current = dpg.get_value("log_box") or ""
            new = current + text + "\n"
            dpg.set_value("log_box", new)
        except Exception:
            # Si no podemos usar la GUI (por ejemplo durante salida), imprimimos por consola
            print(text)


if __name__ == '__main__':
    # Lanza la aplicación GUI
    BLEGuiApp()
