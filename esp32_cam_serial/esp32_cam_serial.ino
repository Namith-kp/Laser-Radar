/*
  ESP32-CAM (AI-Thinker) -> JPEG frames over the USB-TTL cable (no WiFi).

  Wiring (USB-TTL converter  <->  ESP32-CAM):
      5V   -> 5V          GND -> GND
      TX   -> U0R (GPIO3)       RX -> U0T (GPIO1)       (TX and RX are crossed)
  To UPLOAD this sketch: connect GPIO0 to GND, press RESET, upload, then REMOVE the GPIO0-GND wire and press RESET.
  Arduino IDE: Board = "AI Thinker ESP32-CAM", close the Serial Monitor when the tracker runs.

  Every picture is sent as:  A5 5A C3 3C | length (4 bytes, little-endian) | JPEG data
  The tracker reads it with:  python tracker.py --source serial:COM5 ...
*/
#include "esp_camera.h"

// ---------- change these if you like ----------
#define BAUD            2000000   // 2 Mbaud ~ 200 KB/s. If the picture is corrupted try 1000000 or 921600 (and use --cam-baud in the tracker)
#define FRAME_SIZE      FRAMESIZE_QVGA   // QVGA 320x240 (fast).  FRAMESIZE_VGA = 640x480 (slower over serial)
#define JPEG_QUALITY    12        // 10-20; higher number = smaller file = faster
#define MIRROR_HORIZONTAL 1       // 1 = horizontal flip ON (default)
#define FLIP_VERTICAL     0
// -----------------------------------------------

// AI-Thinker pin map
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

void setup() {
  Serial.begin(BAUD);
  Serial.setDebugOutput(false);          // no log text mixed into the picture data

  camera_config_t c;
  c.ledc_channel = LEDC_CHANNEL_0;  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAME_SIZE;
  c.jpeg_quality = JPEG_QUALITY;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;      // always the newest picture -> lowest delay

  if (esp_camera_init(&c) != ESP_OK) { delay(1000); ESP.restart(); }

  // The same default settings every time the board starts (no web page needed)
  sensor_t *s = esp_camera_sensor_get();
  s->set_hmirror(s, MIRROR_HORIZONTAL);
  s->set_vflip(s, FLIP_VERTICAL);
  s->set_whitebal(s, 1);   s->set_awb_gain(s, 1);   s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);  s->set_aec2(s, 1);   s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);  s->set_gainceiling(s, (gainceiling_t)2);   // 8x
  s->set_brightness(s, 0); s->set_contrast(s, 0);   s->set_saturation(s, 1);
  s->set_bpc(s, 1); s->set_wpc(s, 1); s->set_raw_gma(s, 1); s->set_lenc(s, 1);
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  uint32_t n = fb->len;
  const uint8_t head[8] = {0xA5, 0x5A, 0xC3, 0x3C,
                           (uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16), (uint8_t)(n >> 24)};
  Serial.write(head, 8);
  Serial.write(fb->buf, n);              // blocks until sent; the next frame is grabbed fresh afterwards
  esp_camera_fb_return(fb);
}
