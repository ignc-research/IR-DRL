#include <Arduino.h>

void setup() {
  Serial.begin(9600);
}

void loop() {
  int randomNumber = random(1, 101); // generates a random number between 1 and 100
  Serial.println(randomNumber);
  delay(1000); // wait for 1 second
}