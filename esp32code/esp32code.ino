#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// Replace with your network credentials
const char* ssid = "devendra11_fbrtn_2.4";
const char* password = "9805377285";

// Replace with your Django server URL
const char* serverURL = "http://192.168.1.68:8000/receive/";

void setup() {
  Serial.begin(115200);
  
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
  
  // Print accelerometer values to serial monitor
  Serial.print("Accel X: "); Serial.print(a.acceleration.x);
  Serial.print(", Accel Y: "); Serial.print(a.acceleration.y);
  Serial.print(", Accel Z: "); Serial.println(a.acceleration.z);

  // Send accelerometer data to Django server
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverURL);

    http.addHeader("Content-Type", "application/json");
    
    // Prepare JSON data for accelerometer values
    String jsonData = "{\"ax\":\"" + String(a.acceleration.x) + "\",\"ay\":\"" + String(a.acceleration.y) +
                      "\",\"az\":\"" + String(a.acceleration.z) + "\"}";
    
    int httpResponseCode = http.POST(jsonData);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Server Response: " + response);
    } else {
      Serial.println("Error on sending POST: " + String(httpResponseCode));
    }
    
    http.end();
  }
  
  delay(2000);  // Send data every 2 seconds
}
