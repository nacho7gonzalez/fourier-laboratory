#pragma once

// Nombre del dispositivo BLE
#define DEVICE_NAME "GRUPO 1 - XIAO nRF52840 Sense"

// UUIDs de servicios y características
#define ADC_SERVICE_UUID "55c40000-0100-4dd1-be0c-40588193b485"
#define ADC_CHARACTERISTIC_UUID "55c40000-0130-4dd1-be0c-40588193b485"

#define IMU_SERVICE_UUID "55c40000-0200-4dd1-be0c-40588193b485"
#define IMU_CHARACTERISTIC_UUID "55c40000-0230-4dd1-be0c-40588193b485"

#define CONFIG_SERVICE_UUID "55c40000-0300-4dd1-be0c-40588193b485"
#define GAIN_CHARACTERISTIC_UUID "55c40000-0330-4dd1-be0c-40588193b485"
#define FILTRO_CHARACTERISTIC_UUID "55c40000-0340-4dd1-be0c-40588193b485"


// Configuración de buffers
#define ADC_BUFFER_INT16_NUM_SAMPLES    120  // max 122
#define IMU_BUFFER_INT16_NUM_SAMPLES    120  // max 122, debe ser un múltiplo de 3 para poner ejes X, Y, Z

// Frecuencias de muestreo
#define ADC_FREQUENCY_HZ  200
#define IMU_FREQUENCY_HZ  50

//IMU settings
#define IMU_ACCEL_RANGE 4      //Max G force readable.  Can be: 2, 4, 8, 16

//ADC settings
#define ADC_READ_RESOLUTION 12  // bits  (máxima resolución del ADC del nRF52840)   
