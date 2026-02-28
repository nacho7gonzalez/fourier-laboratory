#include "BuffersManager.h"
#include "Config.h"

//Buffers dobles para los datos de ADC e IMU
int16_t adcBuffer0[ADC_BUFFER_INT16_NUM_SAMPLES];
int16_t adcBuffer1[ADC_BUFFER_INT16_NUM_SAMPLES];
int16_t imuBuffer0[IMU_BUFFER_INT16_NUM_SAMPLES];
int16_t imuBuffer1[IMU_BUFFER_INT16_NUM_SAMPLES];

// Banderas para indicar qué buffer se está llenando
bool adcBuffer0AcqActive = true;
bool imuBuffer0AcqActive = true;
// Posición actual en el buffer
uint16_t adcBufferPosition = 0;
uint16_t imuBufferPosition = 0;
// Bandera para indicar cuándo un buffer está lleno y listo para ser enviado
bool adcBufferComplete = false;
bool imuBufferComplete = false;


//******************************************************************************************************
// Buffers - Setup 
//******************************************************************************************************
void setupBuffers(){
  adcBuffer0AcqActive = true;
  imuBuffer0AcqActive = true;
  adcBufferPosition = 0;
  imuBufferPosition = 0;
  adcBufferComplete = false;
  imuBufferComplete = false;
}


void updateAdcBuffer(int16_t sample) {
  if(adcBuffer0AcqActive) {
    adcBuffer0[adcBufferPosition++] = sample;
  } else {
    adcBuffer1[adcBufferPosition++] = sample;
  }

  if(adcBufferPosition >= ADC_BUFFER_INT16_NUM_SAMPLES) {
    adcBuffer0AcqActive = !adcBuffer0AcqActive; // Switch buffers
    adcBufferComplete = true;
    adcBufferPosition = 0;
  }
}       


void updateImuBuffer(int16_t x, int16_t y, int16_t z) {

  if(imuBuffer0AcqActive) {
    imuBuffer0[imuBufferPosition++] = x;
    imuBuffer0[imuBufferPosition++] = y;
    imuBuffer0[imuBufferPosition++] = z;
  } else {
    imuBuffer1[imuBufferPosition++] = x;
    imuBuffer1[imuBufferPosition++] = y;
    imuBuffer1[imuBufferPosition++] = z;
  }

  if(imuBufferPosition >= IMU_BUFFER_INT16_NUM_SAMPLES) {
    imuBuffer0AcqActive = !imuBuffer0AcqActive; // Switch buffers
    imuBufferComplete = true;
    imuBufferPosition = 0;
  }
}   

const void* getAdcBufferToNotify() {
  if(adcBuffer0AcqActive) {
    return (const void *)adcBuffer1;
  } else {
    return (const void *)adcBuffer0;
  }
}

const void* getImuBufferToNotify() {
  if(imuBuffer0AcqActive) {
    return (const void *)imuBuffer1;
  } else {
    return (const void *)imuBuffer0;
  }
}


