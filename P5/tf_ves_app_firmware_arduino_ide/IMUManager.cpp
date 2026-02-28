
#include "IMUManager.h"

#include "Config.h"
#include <LSM6DS3.h>
#include <Wire.h>

//******************************************************************************************************
// Ver: https://github.com/Seeed-Studio/Seeed_Arduino_LSM6DS3/tree/master
//******************************************************************************************************
#define LSM6DS3_ADDR 0x6A  // or 0x6B depending on SA0 pin
LSM6DS3 imu(I2C_MODE, LSM6DS3_ADDR); // I2C address

//******************************************************************************************************
// IMU - Setup
//******************************************************************************************************
bool setupIMU(){

  
  //Over-ride default settings if desired
  imu.settings.gyroEnabled = 1;  //Can be 0 or 1
//  imu.settings.gyroRange = 2000;   //Max deg/s.  Can be: 125, 245, 500, 1000, 2000
//  imu.settings.gyroSampleRate = 833;   //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666
//  imu.settings.gyroBandWidth = 200;  //Hz.  Can be: 50, 100, 200, 400;
  imu.settings.gyroFifoEnabled = 1;  //Set to include gyro in FIFO
//  imu.settings.gyroFifoDecimation = 1;  //set 1 for on /1

  // Disable temperature sensor
  imu.settings.tempEnabled = false;

  imu.settings.accelEnabled = 1;
  imu.settings.accelRange = IMU_ACCEL_RANGE;      //Max G force readable.  Can be: 2, 4, 8, 16
  // SETEO accelSampleRate EL DOBLE DE LO QUE VAMOS A LEER. REVISAR
  imu.settings.accelSampleRate = 104;  //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
  imu.settings.accelBandWidth = 50;  //Hz.  Can be: 50, 100, 200, 400;

  imu.settings.timestampEnabled = 0;        // Disable timestamp feature
  imu.settings.timestampFifoEnabled = 0;    // write timestamp data in FIFO

  imu.settings.accelFifoEnabled = 0;  //Set to include accelerometer in the FIFO
  //imu.settings.accelFifoDecimation = 1;  //set 1 for on /1
  

  //imu.settings.fifoSampleRate = 50; //Hz.  Can be: 10, 25, 50, 100, 200, 400, 800, 1600, 3300, 6600
  //imu.settings.fifoThreshold = IMU_BUFFER_INT16_NUM_SAMPLES*2; 
  imu.settings.fifoModeWord = 0;   //FIFO mode OFF.  Can be:
  //  0 (Bypass mode, FIFO off)
  //  1 (Stop when full)
  //  3 (Continuous during trigger)
  //  4 (Bypass until trigger)
  //  6 (Continous mode)
  
  
  if( imu.begin() != 0 )
  {
    Serial.println("Problem starting IMU");
  }
  else
  {
    Serial.println("IMU started.");
    return true;
  }

  return false;
}

int16_t getRawAccelX() {
  return imu.readRawAccelX();
}
int16_t getRawAccelY() {
  return imu.readRawAccelY();
}
int16_t getRawAccelZ() {
  return imu.readRawAccelZ();
}

// Funciones para obtener los datos del giroscopio
int16_t getRawGyroX() {
    return imu.readRawGyroX();
}

int16_t getRawGyroY() {
    return imu.readRawGyroY();
}

int16_t getRawGyroZ() {
    return imu.readRawGyroZ();
}





