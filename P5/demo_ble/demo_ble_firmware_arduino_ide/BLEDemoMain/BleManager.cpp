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
BLEService configService(CONFIG_SERVICE_UUID);
BLECharacteristic gainCharacteristic(GAIN_CHARACTERISTIC_UUID);

// Nueva característica: POT (para controlar potenciómetro digital)
BLECharacteristic potCharacteristic(POT_CHARACTERISTIC_UUID);

// banderas de estado
bool adcNotificationsEnabled = false;
bool imuNotificationsEnabled = false;
bool connectedFlag = false;     // set by connect callback


//******************************************************************************************************
// MCP4131 (potenciómetro digital) - configuración SPI y funciones
//******************************************************************************************************
// Pines SPI 
#define MOSI_PIN 10
#define SCK_PIN  8
#define CS_PIN   2

// MCP4131-503 datos
#define MAX_RESISTANCE 50000  // 50kΩ
#define WIPER_STEPS 128       // 0..128 (129 posiciones)

// Inicializa los pines / SPI para el potenciómetro
void initPotHardware() {
  // Configura el pin CS como salida e inicializa SPI
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH); // CS en alto (inactivo)
  // SPI.begin() es idempotente
  SPI.begin();
}

// Convierte ohms en valor de wiper y escribe por SPI
void setPot(uint16_t targetOhms) {
  // Recortar a rango válido
  if (targetOhms >= MAX_RESISTANCE) {
    targetOhms = MAX_RESISTANCE;
  }

  // Cálculo del wiper: valor entre 0 y WIPER_STEPS
  float stepRes = (float)MAX_RESISTANCE / (float)WIPER_STEPS;
  int wiper = (int)round((float)targetOhms / stepRes);
  if (wiper < 0) wiper = 0;
  if (wiper > WIPER_STEPS) wiper = WIPER_STEPS;

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
  Serial.print(wiper * stepRes);
  Serial.println(" ohm");
}


//******************************************************************************************************
// BLE - Setuup del nombre del dispositivo
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
  Bluefruit.begin();
  Bluefruit.setTxPower(0);
  Bluefruit.setConnLedInterval(50); //ms
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);
}

// bool imuSampleReady = false;
void connect_callback(uint16_t conn_handle)
{
  connectedFlag = false;
  
  Serial.print("【connect_callback】 conn_Handle : ");
  Serial.println(conn_handle, HEX);

  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);

  // request to chamge parameters
  connection->requestPHY();                     // PHY 2MHz (2Mbit/sec moduration) 1 --> 2
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

  // Incluir ambos servicios (ADC y CONFIG) en el advertising
  Bluefruit.Advertising.addService(adcService);
  Bluefruit.Advertising.addService(configService);

  // (opcional) Nombre en scan response
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
  
  configService.begin();
  gainCharacteristic.setProperties(CHR_PROPS_WRITE | CHR_PROPS_READ);
  gainCharacteristic.setFixedLen(2);
  String gainDesc = String("Gain (int16_t)");
  gainCharacteristic.setUserDescriptor(gainDesc.c_str());
  gainCharacteristic.begin();
  gainCharacteristic.setWriteCallback(gain_write_callback);

  // Inicializar hardware del potenciómetro digital (CS pin, SPI)
  initPotHardware();

  // --- Nueva característica de POT ---
  potCharacteristic.setProperties(CHR_PROPS_WRITE | CHR_PROPS_READ);
  potCharacteristic.setFixedLen(2);
  String potDesc = String("Pot (int16_t - ohms)");
  potCharacteristic.setUserDescriptor(potDesc.c_str());
  potCharacteristic.begin();
  potCharacteristic.setWriteCallback(pot_write_callback);
  
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
    // You may want to access the instance here if needed
    adcNotificationsEnabled = true;
  } else {
    Serial.println("¡Central se ha desuscrito de las notificaciones ADC!");
    adcNotificationsEnabled = false;
  }
}

///******************************************************************************************************
// Callback de escritura en característica de ganancia
//******************************************************************************************************
// Cuando el cliente escribe en la característica de ganancia, se lee el valor y se muestra por el puerto serie
void gain_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
  // Protecciones básicas
  if (len < 2) {
    Serial.println("gain_write_callback: len < 2");
    return;
  }

  // Interpretar little-endian (evita problemas de alineamiento)
  int16_t int16_value = (int16_t)(data[0] | (data[1] << 8));
  Serial.print("Valor de ganancia recibido del Central: ");
  Serial.print(int16_value);
  Serial.println();
  
}


//******************************************************************************************************
// Callback de escritura en característica de POT (ahora controla MCP4131)
//******************************************************************************************************
void pot_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
  // Asumimos que el cliente nos manda un int16_t little-endian con el valor de ohms deseado
  if (len < 2) {
    Serial.println("pot_write_callback: len < 2");
    return;
  }

  int16_t target = (int16_t)(data[0] | (data[1] << 8));
  if (target < 0) {
    Serial.print("pot_write_callback: valor negativo recibido: ");
    Serial.println(target);
    return;
  }

  Serial.print("Valor de pot recibido del Central (ohms): ");
  Serial.println(target);

  // Llamar función que maneja el MCP4131
  setPot((uint16_t) target);
}



 //******************************************************************************************************
// Notificación de datos ADC al cliente
//******************************************************************************************************
void notifyAdc(int16_t* data) {
  if (connectedFlag && adcNotificationsEnabled) {
    adcDataCharacteristic.notify((uint8_t*)data, ADC_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
  }
}
