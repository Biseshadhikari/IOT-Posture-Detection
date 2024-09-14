#include <WiFi.h>
#include <HTTPClient.h>

const char *ssid = "your-SSID";
const char *password = "your-PASSWORD";

void setup()
{
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(1000);
        Serial.println("Connecting...");
    }

    HTTPClient http;
    http.begin("http://your-django-server/receive/");
    http.addHeader("Content-Type", "application/json");

    // Prepare the JSON payload
    String jsonData = "{\"param1\": 12.3, \"param2\": 45.6, \"param3\": 78.9, \"param4\": 23.4, \"param5\": 56.7}";

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0)
    {
        String response = http.getString();
        Serial.println(response);
    }
    else
    {
        Serial.println("Error in sending POST request");
    }

    http.end();
}

void loop()
{
    // Do nothing in the loop
}
