#pragma once
#include <stdint.h>

// Los tamaños de los buffers están definidos en Config.h
#include "Config.h"

extern bool adcBufferComplete;
extern bool imuBufferComplete;

extern int16_t adcBuffer0[ADC_BUFFER_INT16_NUM_SAMPLES];
extern int16_t adcBuffer1[ADC_BUFFER_INT16_NUM_SAMPLES];
extern int16_t imuBuffer0[IMU_BUFFER_INT16_NUM_SAMPLES];
extern int16_t imuBuffer1[IMU_BUFFER_INT16_NUM_SAMPLES];
extern bool adcBuffer0AcqActive;
extern bool imuBuffer0AcqActive;

void setupBuffers();
void updateAdcBuffer(int16_t sample);
void updateImuBuffer(int16_t x, int16_t y, int16_t z);
const void * getAdcBufferToNotify();
const void * getImuBufferToNotify();

