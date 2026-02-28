#include "LedsManager.h"
#include <bluefruit.h>
//******************************************************************************************************
// LEDS 
//******************************************************************************************************


LedsManager::LedsManager(){
  // Pin initialization
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT); 

  turnOffLeds();
}

void LedsManager::indicateError(){
  // Blink red LED to indicate error
  digitalWrite(LED_RED, LOW);
  delay(100);
  digitalWrite(LED_RED, HIGH);
  delay(100);
}

void LedsManager::toggleGreenLed(){
  static bool ledState = false;
  ledState = !ledState;
  digitalWrite(LED_GREEN, ledState ? LOW : HIGH);
}
void LedsManager::toggleBlueLed(){
  static bool ledState = false;
  ledState = !ledState;
  digitalWrite(LED_BLUE, ledState ? LOW : HIGH);
}   
void LedsManager::toggleRedLed(){
  static bool ledState = false;
  ledState = !ledState;
  digitalWrite(LED_RED, ledState ? LOW : HIGH);
}
void LedsManager::turnOffLeds(){
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_BLUE, HIGH);
}
void LedsManager::turnOnLeds(){
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_BLUE, LOW);
}
