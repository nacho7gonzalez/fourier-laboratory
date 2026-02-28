#pragma once
#include <Arduino.h>


#define HW_TIMER_INTERVAL_MS      1


//-----Flags to signal when to read sensors -----
extern bool imuSampleReady;
extern bool adcSampleReady;

void setupTimer(void (*adcCallback)(), void (*imuCallback)());
void TimerHandler();
void adcTimerRoutine();
void imuTimerRoutine();



