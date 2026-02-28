
#pragma once
#include <bluefruit.h>

//*****************************************************************************************************
// BLE basado en la biblioteca Bluefruit
// Ver: https://github.com/adafruit/Adafruit_nRF52_Arduino/tree/master/libraries/Bluefruit52Lib
//*****************************************************************************************************



// SOLO UN GENERADOR DE UUIDs
// 55c40000-xxxx-4dd1-be0c-40588193b485   donde xxxx es el valor pasado como argumento
// ejemplo: 55c40000-0002-4dd1-be0c-40588193b485 para el valor 0x0002
#define UUID_GENERATOR(val) (const uint8_t[]) { \
    0x85, 0xB4, 0x93, 0x81, 0x58, 0x40, 0x0C, 0xBE, \
    0xD1, 0x4D, (uint8_t)(val & 0xff), (uint8_t)(val >> 8), 0x00, 0x00, 0xC4, 0x55 }



extern bool connectedFlag;     // set by connect callback
extern bool adcNotificationsEnabled;
extern bool imuNotificationsEnabled;

				        
void setupBLEConnection();
void setupBLEServices();
void setupBLEAdvertising();
void setupBLEDeviceName(const char* name);
void notifyAdc(int16_t* data);

void connect_callback(uint16_t conn_handle);
void disconnect_callback(uint16_t conn_handle, uint8_t reason);
void gain_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len);
void pot_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len);
void adc_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value);
void imu_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value);
