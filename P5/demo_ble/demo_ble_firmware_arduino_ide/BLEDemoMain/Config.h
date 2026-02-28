#pragma once

// Nombre del dispositivo BLE
#define DEVICE_NAME "GRUPO 1 - XIAO nRF52840 Sense"

// UUIDs de servicios y características
#define ADC_SERVICE_UUID "55c40000-0100-4dd1-be0c-40588193b485"
#define ADC_CHARACTERISTIC_UUID "55c40000-0130-4dd1-be0c-40588193b485"

#define CONFIG_SERVICE_UUID "55c40000-0300-4dd1-be0c-40588193b485"
#define GAIN_CHARACTERISTIC_UUID "55c40000-0330-4dd1-be0c-40588193b485"
#define POT_CHARACTERISTIC_UUID "55c40000-0340-4dd1-be0c-40588193b485"
// Configuración de buffers
#define ADC_BUFFER_INT16_NUM_SAMPLES    10  // Cantidad de muestras de tipo int16_t, NO PASAR DE 122


