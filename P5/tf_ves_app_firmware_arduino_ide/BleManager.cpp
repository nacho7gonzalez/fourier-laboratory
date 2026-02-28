#include "BleManager.h"
#include "Config.h"
#include <Arduino.h>
#include <SPI.h>

//*****************************************************************************************************
// BLE con biblioteca Bluefruit
// Ver: https://learn.adafruit.com/introducing-the-adafruit-nrf52840-feather/bluefruit-nrf52-api
//*****************************************************************************************************

// Características y servicios BLE
BLEService adcService(ADC_SERVICE_UUID);
BLECharacteristic adcDataCharacteristic(ADC_CHARACTERISTIC_UUID);
BLEService imuService(IMU_SERVICE_UUID);
BLECharacteristic imuDataCharacteristic(IMU_CHARACTERISTIC_UUID);

BLEService configService(CONFIG_SERVICE_UUID);
BLECharacteristic gainCharacteristic(GAIN_CHARACTERISTIC_UUID);
// <-- Implementación de filtroCharacteristic (1 byte): 0 = OFF, 1 = ON
BLECharacteristic filtroCharacteristic(FILTRO_CHARACTERISTIC_UUID);

// banderas de estado
bool adcNotificationsEnabled = false;
bool imuNotificationsEnabled = false;
bool connectedFlag = false;     // set by connect callback

// Bandera para activar/desactivar filtro (valor por defecto true)
bool filtroEnabled = true;

//******************************************************************************************************
// MCP4131 (potenciómetro digital) - configuración SPI y funciones
//******************************************************************************************************
// Pines SPI 
#define MOSI_PIN 10
#define MISO_PIN 12 // añadido para completar la definición (no usado explícitamente en este fichero)
#define SCK_PIN  8
#define CS_PIN   2

// MCP4131-503 datos
#define MAX_RESISTANCE 50000  // 50kΩ
#define WIPER_STEPS 128       // número de posiciones (0..127 -> 128 posiciones)

#define WIPER_MAX (WIPER_STEPS - 1) // wiper máximo válido (127)

// Prototipos (si tu proyecto usa declaraciones en headers, mantenelos; aquí por seguridad)
void connect_callback(uint16_t conn_handle);
void disconnect_callback(uint16_t conn_handle, uint8_t reason);
void adc_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value);
void imu_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value);
void gain_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len);

// *****************************************************************************************************
// Callback de escritura en característica filtro (definida antes de usarla)
// *****************************************************************************************************
void filtro_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
  (void) conn_hdl;
  (void) chr;

  if (len < 1) {
    Serial.println("filtro_write_callback: data too short");
    return;
  }

  uint8_t v = data[0];
  if (v == 0) {
    filtroEnabled = false;
    Serial.println("Filtro desactivado via BLE (0)");
  } else {
    filtroEnabled = true;
    Serial.println("Filtro activado via BLE (1)");
  }

  // Actualizo el valor almacenado en la característica para que una lectura devuelva el estado actual
  filtroCharacteristic.write8(filtroEnabled ? 1 : 0);
}

// *****************************************************************************************************
// Convierte ohms en valor de wiper y escribe por SPI
// *****************************************************************************************************
void setPot(uint16_t targetOhms) {

  // Ajusto la resistencia para el otro lado
  targetOhms = 50000 - targetOhms;

  // Recortar a rango válido
  if (targetOhms >= MAX_RESISTANCE) {
    targetOhms = MAX_RESISTANCE;
  }

  // Cálculo del wiper: valor entre 0 y WIPER_MAX
  float stepRes = (float)MAX_RESISTANCE / (float)WIPER_STEPS; // Resistencia aproximada por paso
  int wiper = (int)round((float)targetOhms / stepRes);
  if (wiper < 0) wiper = 0;
  if (wiper > WIPER_MAX) wiper = WIPER_MAX;

  // Realiza la transferencia (comando + dato)
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  digitalWrite(CS_PIN, LOW);
  // Para MCP4131: primer byte = comando (0x00 suele usarse para escribir wiper)
  SPI.transfer(0x00);
  SPI.transfer((uint8_t)wiper);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();

  // Debug
  Serial.print("setPot(): wiper=");
  Serial.print(wiper);
  Serial.print(" -> approx ");
  Serial.print((int)(50000 - (wiper * stepRes)));
  Serial.println(" ohm");
}

// Inicializa los pines / SPI para el potenciómetro
void initPotHardware() {
  // Configura el pin CS como salida e inicializa SPI
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH); // CS en alto (inactivo)

  // Inicializa SPI hardware (idempotente)
  SPI.begin();

  // Ajuste por defecto: 10kOhm al arrancar
  setPot(10000); // <-- valor por defecto requerido
}

//******************************************************************************************************
// BLE - Setup del nombre del dispositivo
//******************************************************************************************************
void setupBLEDeviceName(const char* name) { 
  Bluefruit.setName(name); 
}

//******************************************************************************************************
// BLE - Setup de conexión
//******************************************************************************************************
// Parámetros de conexión
#define CONN_PARAM  6         // connection interval *1.25mS, min 6
#define DATA_NUM    244       // max 244

void setupBLEConnection(){
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.configUuid128Count(15);

  // Inicializo SPI/pot antes de arrancar BLE para asegurar que el pot quede en 10k al boot.
  initPotHardware();

  Bluefruit.begin();
  Bluefruit.setTxPower(0);
  Bluefruit.setConnLedInterval(50); //ms
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);
}


void connect_callback(uint16_t conn_handle)
{
  connectedFlag = false;
  
  Serial.print("【connect_callback】 conn_Handle : ");
  Serial.println(conn_handle, HEX);

  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  // request to change parameters
  connection->requestPHY();                     // PHY 2MHz (2Mbit/sec modulation) 1 --> 2
  delay(1000);                                  // delay a bit for request to complete
  connection->requestDataLengthUpdate();        // data length  27 --> 251

  connection->requestMtuExchange(DATA_NUM + 3); // MTU 23 --> 247
  connection->requestConnectionParameter(CONN_PARAM);   // connection interval (*1.25mS)   
  delay(1000);                                  // delay a bit for request to complete  
  
  Serial.println();
  Serial.print("PHY ----------> "); Serial.println(connection->getPHY());
  Serial.print("Data length --> "); Serial.println(connection->getDataLength());
  Serial.print("MTU ----------> "); Serial.println(connection->getMtu());
  Serial.print("Interval -----> "); Serial.println(connection->getConnectionInterval());      

  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));
  Serial.print("【connect_callback】 Connected to ");
  Serial.println(central_name);

  connectedFlag = true;
}

//***********************************************************************************************
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  Serial.print("【disconnect_callback】 reason = 0x");
  Serial.println(reason, HEX);
  connectedFlag = false;
}

//******************************************************************************************************
// BLE - Setup de Advertising
//******************************************************************************************************

void setupBLEAdvertising(){
  
  // Advertisement Settings
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.ScanResponse.addName();
  
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setIntervalMS(20, 153);     // fast mode 20mS, slow mode 153mS
  Bluefruit.Advertising.setFastTimeout(30);         // fast mode 30 sec
  Bluefruit.Advertising.start(0);                   // 0 = Don't stop advertising after n seconds

}

//******************************************************************************************************
// BLE - Setup de servicios y características
//******************************************************************************************************

void setupBLEServices(){
 
  adcService.begin();
  adcDataCharacteristic.setProperties(CHR_PROPS_NOTIFY);
  adcDataCharacteristic.setFixedLen(ADC_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
  adcDataCharacteristic.setCccdWriteCallback(adc_cccd_callback);
  String desc = String("ADC Data") + String(", ") +  String(ADC_BUFFER_INT16_NUM_SAMPLES) + " samples" + " (int16_t)";
  adcDataCharacteristic.setUserDescriptor(desc.c_str());
  adcDataCharacteristic.begin();

  imuService.begin();
  imuDataCharacteristic.setProperties(CHR_PROPS_NOTIFY);
  imuDataCharacteristic.setFixedLen(IMU_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
  imuDataCharacteristic.setCccdWriteCallback(imu_cccd_callback);
  String imuDesc = String("IMU Data") + String(", ") +  String(IMU_BUFFER_INT16_NUM_SAMPLES) + " samples" + " (int16_t)";
  imuDataCharacteristic.setUserDescriptor(imuDesc.c_str());
  imuDataCharacteristic.begin();
  
  configService.begin();
  gainCharacteristic.setProperties(CHR_PROPS_WRITE | CHR_PROPS_READ);
  gainCharacteristic.setFixedLen(2);
  String gainDesc = String("Gain (int16_t)");
  gainCharacteristic.setUserDescriptor(gainDesc.c_str());
  gainCharacteristic.begin();
  gainCharacteristic.setWriteCallback(gain_write_callback);

  // ---------------------------
  // filtroCharacteristic setup
  // ---------------------------
  filtroCharacteristic.setProperties(CHR_PROPS_WRITE | CHR_PROPS_READ);
  filtroCharacteristic.setFixedLen(1);
  filtroCharacteristic.setUserDescriptor("Filtro ON/OFF (1 byte: 0=OFF,1=ON)");
  filtroCharacteristic.begin();
  filtroCharacteristic.setWriteCallback(filtro_write_callback);

  // Inicializo el valor leído por defecto en la característica (para lecturas desde el Central)
  filtroCharacteristic.write8(filtroEnabled ? 1 : 0);
}

//******************************************************************************************************
// Callback de suscripción/desuscripción a notificaciones ADC
//******************************************************************************************************
// Cuando el cliente se suscribe o desuscribe a las notificaciones de la característica ADC
// se cambia la bandera adc_notifications_enabled
void adc_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value) {
  (void) cccd_value;
  if (chr->notifyEnabled(conn_hdl)) {
    Serial.println("¡Central se ha suscrito a las notificaciones ADC!");
    adcNotificationsEnabled = true;
  } else {
    Serial.println("¡Central se ha desuscrito de las notificaciones ADC!");
    adcNotificationsEnabled = false;
  }
}

//******************************************************************************************************
// Callback de suscripción/desuscripción a notificaciones IMU
//******************************************************************************************************
// Cuando el cliente se suscribe o desuscribe a las notificaciones de la característica IMU
// se cambia la bandera imu_notifications_enabled
void imu_cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value) {
  (void) cccd_value; // cccd_value is not needed with notifyEnabled()

  if (chr->notifyEnabled(conn_hdl)) {
    Serial.println("Central has subscribed to IMU notifications!");
    imuNotificationsEnabled = true;
  } else {
    Serial.println("Central has unsubscribed from IMU notifications.");
    imuNotificationsEnabled = false;
  }
}

///******************************************************************************************************
// Callback de escritura en característica de ganancia
//******************************************************************************************************
// --- Modificar gain_write_callback para aplicar el potenciómetro
void gain_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
  (void) conn_hdl;
  (void) chr;

  if (len < 2) {
    Serial.println("gain_write_callback: data too short");
    return;
  }

  int16_t raw = (int16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
  Serial.print("gain_write_callback: raw int16 value = ");
  Serial.println(raw);

  int32_t desiredOhm = (int32_t) raw;
  if (desiredOhm < 0) desiredOhm = 0;
  if (desiredOhm > MAX_RESISTANCE) desiredOhm = MAX_RESISTANCE;

  // Aplicar al potenciómetro (llamada directa)
  setPot((uint16_t)desiredOhm);
}

//******************************************************************************************************
// Notificación de datos ADC al cliente
//******************************************************************************************************
void notifyAdc(int16_t* data) {
  if (connectedFlag && adcNotificationsEnabled) {
    adcDataCharacteristic.notify((uint8_t*)data, ADC_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
  }
}
