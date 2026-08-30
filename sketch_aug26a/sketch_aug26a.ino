const int buzzerPin = 8;

void setup() {
  Serial.begin(9600);

  pinMode(buzzerPin, OUTPUT);
}

void loop() {

  if (Serial.available()) {

    String command = Serial.readStringUntil('\n');

    if (command == "BUZZER") {

      tone(buzzerPin, 1000);
      delay(1000);
      noTone(buzzerPin);
    }
  }
}
