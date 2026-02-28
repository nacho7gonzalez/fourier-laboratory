
#pragma once
#include <stdint.h>

class LedsManager {
public:
    LedsManager();
    ~LedsManager(){};

    void indicateError();
    void toggleGreenLed();
    void toggleBlueLed();
    void toggleRedLed();
    void turnOffLeds();
    void turnOnLeds();
private:
    // Add private members here
};
