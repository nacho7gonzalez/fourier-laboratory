#pragma once
#include <stdint.h>

bool setupIMU();
int16_t getRawAccelX(); 
int16_t getRawAccelY();
int16_t getRawAccelZ();


// Nuevas declaraciones para el Giroscopio
int16_t getRawGyroX();
int16_t getRawGyroY();
int16_t getRawGyroZ();
