//----------------------------------------------------------------------------------------------
// Taller Fourier Vestible - TF-VES
// Board : Seeed XIAO nRF52840 Sense o  Seeed XIAO nRF52840 Sense Plus 
//----------------------------------------------------------------------------------------------

#include <Adafruit_TinyUSB.h> 
#include <bluefruit.h>          

#include <Wire.h>
#include <math.h>   // roundf

#include "Config.h"
#include "TimerManager.h"
#include "IMUManager.h"
#include "BuffersManager.h"
#include "BleManager.h"
#include "LedsManager.h"

// Declaración de la variable definida en BleManager.cpp
extern bool filtroEnabled; // 0 = OFF, 1 = ON

// =========================
//  Filtro Butterworth HP2
// =========================
// Coeficientes calculados previamente
// b0 = 0.98238544; b1 = -1.96477088; b2 = 0.98238544;
// a1 = -1.96446058; a2 = 0.96508117;

typedef struct {
    float x1; // x[n-1]
    float x2; // x[n-2]
    float y1; // y[n-1]
    float y2; // y[n-2]
} ButterHPState;

void resetButterHP(ButterHPState *st){
    st->x1 = 0.0f;
    st->x2 = 0.0f;
    st->y1 = 0.0f;
    st->y2 = 0.0f;
}

float butterHP_apply(float x, ButterHPState *st){
    const float b0 =  0.98238544f;
    const float b1 = -1.96477088f;
    const float b2 =  0.98238544f;
    const float a1 = -1.96446058f;
    const float a2 =  0.96508117f;

    // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    float y = b0 * x + b1 * st->x1 + b2 * st->x2 - a1 * st->y1 - a2 * st->y2;

    // desplazar estados
    st->x2 = st->x1;
    st->x1 = x;

    st->y2 = st->y1;
    st->y1 = y;

    return y;
}

// Estados del filtro para cada eje del acelerómetro
ButterHPState hp_ax;
ButterHPState hp_ay;
ButterHPState hp_az;


//*****************************************************************************************************
// Callbacks llamado por el Interrupt Timer 
//*****************************************************************************************************
void imuSampleReadyCallback(){
    imuSampleReady = true; // Setear la bandera
}

void adcSampleReadyCallback(){
    if (adcNotificationsEnabled){
      updateAdcBuffer(analogRead(A0));
      adcSampleReady = false; // Resetear la bandera
    }
}

//*****************************************************************************************************
// SETUP y LOOP
// 
//*****************************************************************************************************
void setup()
{
   // Inicialización de comunicación serie
  Serial.begin(115200);
  delay(2000);
 
  Serial.println("-------------------------------------");
  Serial.println("Inicialización.......................");

  analogReadResolution(ADC_READ_RESOLUTION); // Configurar resolución ADC
  
  setupIMU();
   
  setupBLEConnection();  
  setupBLEDeviceName(DEVICE_NAME);  // debe ir luego de setupBLEConnection
  setupBLEServices();

  // Inicializar estados del filtro (todo en 0)
  resetButterHP(&hp_ax);
  resetButterHP(&hp_ay);
  resetButterHP(&hp_az);

  setupTimer( adcSampleReadyCallback, imuSampleReadyCallback); 
  setupBLEAdvertising();
   
  Serial.println("Fin de inicialización");

  // Connecting to Central
  Serial.println("Conectando a Central ................");
  while(!Bluefruit.connected()) {
    Serial.print("."); delay(100);
  }
  Serial.println();
  Serial.println("Conectado a Central");
  delay(5000);
}

 //******************************************************************************************************

void loop()
{
  if (connectedFlag == true)        // Si hay una conexión al central
  {
    if (imuSampleReady){
      // Leer los valores del IMU (raw int16)
      int16_t ax_raw = getRawAccelX();
      int16_t ay_raw = getRawAccelY();
      int16_t az_raw = getRawAccelZ();

      int16_t gx = getRawGyroX();
      int16_t gy = getRawGyroY();
      int16_t gz = getRawGyroZ();

      int16_t ax, ay, az;

      if (filtroEnabled) {
        // --- Filtrar ax, ay, az ---
        float ax_f = (float) ax_raw;
        float ay_f = (float) ay_raw;
        float az_f = (float) az_raw;

        float ax_filt = butterHP_apply(ax_f, &hp_ax);
        float ay_filt = butterHP_apply(ay_f, &hp_ay);
        float az_filt = butterHP_apply(az_f, &hp_az);

        ax = (int16_t) roundf(ax_filt);
        ay = (int16_t) roundf(ay_filt);
        az = (int16_t) roundf(az_filt);
      } else {
        // Filtro desactivado: pasar valores crudos tal cual
        ax = ax_raw;
        ay = ay_raw;
        az = az_raw;

        // Debemos "respetar" el estado del filtro si más tarde se activa:
        // opcional: mantener estados del IIR o resetear. Aquí NO los reseteamos,
        // para evitar transitorios abruptos cuando se encienda el filtro.
        //
        // Si preferís resetear al desactivar:
        // resetButterHP(&hp_ax); resetButterHP(&hp_ay); resetButterHP(&hp_az);
      }

      // Llamar al update con las muestras (filtradas o crudas según bandera)
      updateImuBuffer(ax, ay, az);
      updateImuBuffer(gx, gy, gz); // Mando los datos del gyro al otro buffer

      imuSampleReady = false; // Resetear la bandera
    }

    if(adcNotificationsEnabled && adcBufferComplete){
      adcDataCharacteristic.notify(getAdcBufferToNotify(), ADC_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
      adcBufferComplete = false;
    }

    if(imuNotificationsEnabled && imuBufferComplete){
      imuDataCharacteristic.notify(getImuBufferToNotify(), IMU_BUFFER_INT16_NUM_SAMPLES * sizeof(int16_t));
      imuBufferComplete = false;
    }

    delay(10); // Esperar un poco para no saturar el loop
  }
}
