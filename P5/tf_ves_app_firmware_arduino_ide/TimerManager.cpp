#include "Config.h"
#include "TimerManager.h"
#include "NRF52TimerInterrupt.h"
#include "NRF52_ISR_Timer.h"


#define ADC_TIMER_INTERVAL_MS  1000/ADC_FREQUENCY_HZ
#define IMU_TIMER_INTERVAL_MS  1000/IMU_FREQUENCY_HZ


// Ver: https://github.com/khoih-prog/NRF52_TimerInterrupt


NRF52Timer ITimer(NRF_TIMER_3);
NRF52_ISR_Timer ISR_Timer;

// Initialize flags
bool adcSampleReady = false;
bool imuSampleReady = false;


//******************************************************************************************************
void setupTimer(void (*adcCallback)(), void (*imuCallback)()) {
  Serial.print(F("\nStarting TimerInterrupt on "));
  Serial.println(BOARD_NAME);
  Serial.println(NRF52_TIMER_INTERRUPT_VERSION);
  Serial.print(F("CPU Frequency = "));
  Serial.print(F_CPU / 1000000);
  Serial.println(F(" MHz"));

  if (ITimer.attachInterruptInterval(1 * 1000, TimerHandler)) {
    Serial.print(F("Starting ITimer OK, millis() = "));
    Serial.println(millis());
  } else {
    Serial.println(F("Can't set ITimer. Select another freq. or timer"));
  }

  ISR_Timer.setInterval(ADC_TIMER_INTERVAL_MS,  adcCallback);
  ISR_Timer.setInterval(IMU_TIMER_INTERVAL_MS,  imuCallback);
  //ISR_Timer.setInterval(CHECK_BUFFER_FULL_TIMER_INTERVAL_MS, checkBufferFullRoutine);
}

void TimerHandler() {
  ISR_Timer.run();
}




