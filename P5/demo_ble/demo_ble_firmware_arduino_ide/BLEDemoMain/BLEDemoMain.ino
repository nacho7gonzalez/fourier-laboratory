//----------------------------------------------------------------------------------------------
// 
// Board : Seeed nRF52 Boards / Seeed XIAO nRF52840 Sense 
//----------------------------------------------------------------------------------------------
#include <Adafruit_TinyUSB.h> 

#include "BleManager.h"
#include "Config.h"

//*****************************************************************************************************
// SETUP y LOOP
// 
//*****************************************************************************************************


void setup()
{
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("-------------------------------------");
  Serial.println("Inicialización BLE .......................");
  
  setupBLEConnection();
  setupBLEDeviceName(DEVICE_NAME);  // debe ir luego de setupBLEConnection
  setupBLEServices();
  setupBLEAdvertising();

  Serial.println("Fin de la inicialización BLE");
  Serial.println("-------------------------------------");
  Serial.println("Esperando conexión de un dispositivo Central ................");
  while(!Bluefruit.connected()) {
    Serial.print("."); delay(100);
  }
  Serial.println();
  Serial.println("Conectado a Central");
  delay(2000);
}

//******************************************************************************************************
  
void loop()
{
  if (connectedFlag) {
    
    // Si Central se ha suscrito a las notificaciones ADC
    // se envían datos simulados cada segundo
    // (en la aplicación real, los datos van a ser muestras provenientes del ADC)
    if(adcNotificationsEnabled){
      int16_t buffer[ADC_BUFFER_INT16_NUM_SAMPLES];
      // Llenar el buffer con datos ADC simulados (aquí simplemente usamos un contador)
      static int16_t counter = 0; //static: para que mantenga su valor entre llamadas  
      for (int i = 0; i < ADC_BUFFER_INT16_NUM_SAMPLES; i++) {
        buffer[i] = counter++;
      }
      // notificar 
      notifyAdc(buffer);
    }
  }
  
  delay(1000); 
}
