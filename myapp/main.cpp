#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>

Adafruit_MPU6050 mpu;

// Replace with your network credentials
const char* ssid = "Display";
const char* password = "aaaabbbb";
 
// Replace with your Django server URL
const char* serverURL = "http://10.36.131.101:8000/receive/";

// Define buzzer pin (GPIO 5)
const int buzzerPin = 5;

void setup() {
  Serial.begin(115200);

  // Set buzzer pin as output
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);  // Ensure buzzer is off initially

  // Buzzer startup signal
  delay(1000);
  digitalWrite(buzzerPin, HIGH);
  delay(500);
  digitalWrite(buzzerPin, LOW); 

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("MPU6050 initialized");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print("Accel X: "); Serial.print(a.acceleration.x);
  Serial.print(", Accel Y: "); Serial.print(a.acceleration.y);
  Serial.print(", Accel Z: "); Serial.println(a.acceleration.z);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    // Proper JSON format without quotes for numbers
    String jsonData = "{\"ax\":" + String(a.acceleration.x, 3) +
                      ",\"ay\":" + String(a.acceleration.y, 3) +
                      ",\"az\":" + String(a.acceleration.z, 3) + "}";

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Server Response: " + response);

      // Parse JSON using ArduinoJson
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, response);

      if (error) {
        Serial.print("JSON parse error: ");
        Serial.println(error.c_str());
      } else {
        bool alert = doc["alert"];
        if (alert) {
          Serial.println("ALERT RECEIVED: Activating buzzer");
          digitalWrite(buzzerPin, HIGH);  // Turn buzzer ON
          delay(500);                     // Buzz duration
          digitalWrite(buzzerPin, LOW);   // Turn buzzer OFF
        }
      }

    } else {
      Serial.print("Error in POST: ");
      Serial.println(httpResponseCode);
    }

    http.end();
  } else {
    Serial.println("WiFi not connected");
  }

  delay(500);  // Wait before sending next reading
}
