/*
 * ══════════════════════════════════════════════════════════════════
 *  ESP32 Traffic Light Controller
 * ══════════════════════════════════════════════════════════════════
 *
 *  Receives serial commands from Python AI Traffic Control System
 *  and drives 4-way traffic light LEDs.
 *
 *  Serial Protocol:
 *    Input:  "N\n" or "E\n" or "S\n" or "W\n"
 *            → Sets selected lane GREEN, all others RED
 *            → Runs for DEFAULT_GREEN_TIME, then YELLOW, then RED
 *            → Sends "READY\n" when done
 *
 *    Alternative Input: "N:10,E:15,S:8,W:12\n"
 *            → Runs each lane GREEN sequentially with specified times
 *
 *  Pin Configuration:
 *    North: R=22, Y=23, G=25
 *    South: R=18, Y=19, G=21
 *    East:  R=33, Y=4,  G=5
 *    West:  R=26, Y=27, G=32
 *
 *  Baud Rate: 115200
 *
 *  Uses non-blocking millis() logic throughout.
 *
 *  Author: AI Traffic Control Project
 */

// ── Pin Definitions ──────────────────────────────────────────────

// North
#define NR 22
#define NY 23
#define NG 25

// South
#define SR 18
#define SY 19
#define SG 21

// East
#define ER 33
#define EY 4
#define EG 5

// West
#define WR 26
#define WY 27
#define WG 32

// ── Timing Configuration ─────────────────────────────────────────

#define DEFAULT_GREEN_TIME  15000   // 15 seconds (milliseconds)
#define YELLOW_TIME         3000    // 3 seconds
#define ALL_RED_TIME        1000    // 1 second pause between phases
#define MIN_GREEN_TIME      5000    // 5 seconds minimum
#define MAX_GREEN_TIME      45000   // 45 seconds maximum

// ── State Machine ────────────────────────────────────────────────

enum State {
  STATE_IDLE,            // Waiting for command
  STATE_ALL_RED,         // Brief all-red before green
  STATE_GREEN,           // Green phase active
  STATE_YELLOW,          // Yellow transition
  STATE_POST_RED,        // Brief all-red after yellow
  STATE_SEQUENCE_NEXT,   // Move to next lane in sequence
};

State currentState = STATE_IDLE;

// ── Lane Data ────────────────────────────────────────────────────

struct Lane {
  const char name;
  uint8_t pinR, pinY, pinG;
};

Lane lanes[4] = {
  {'N', NR, NY, NG},
  {'E', ER, EY, EG},
  {'S', SR, SY, SG},
  {'W', WR, WY, WG},
};

// ── Runtime Variables ────────────────────────────────────────────

unsigned long stateStartTime = 0;    // millis() when current state began
unsigned long greenDuration = DEFAULT_GREEN_TIME;

// For single-lane priority mode
int activeLane = -1;                 // 0=N, 1=E, 2=S, 3=W

// For sequence mode (all lanes with individual timings)
bool sequenceMode = false;
unsigned long laneTimes[4] = {0, 0, 0, 0};
int sequenceIndex = 0;

// Serial input buffer
String inputBuffer = "";

// ═══════════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);

  // Initialize all traffic light pins as OUTPUT
  for (int i = 0; i < 4; i++) {
    pinMode(lanes[i].pinR, OUTPUT);
    pinMode(lanes[i].pinY, OUTPUT);
    pinMode(lanes[i].pinG, OUTPUT);
  }

  // Start with all RED
  setAllRed();

  Serial.println("ESP32 Traffic Controller v2.0");
  Serial.println("Waiting for commands...");
  Serial.println("READY");
}

// ═══════════════════════════════════════════════════════════════════
//  MAIN LOOP (non-blocking state machine)
// ═══════════════════════════════════════════════════════════════════

void loop() {
  // Check for serial input
  checkSerial();

  // Run state machine
  unsigned long now = millis();
  unsigned long elapsed = now - stateStartTime;

  switch (currentState) {
    case STATE_IDLE:
      // Do nothing — waiting for command
      break;

    case STATE_ALL_RED:
      // Brief all-red pause before turning green
      if (elapsed >= ALL_RED_TIME) {
        // Turn active lane green
        if (sequenceMode) {
          activeLane = sequenceIndex;
          greenDuration = laneTimes[sequenceIndex];
        }
        if (activeLane >= 0 && activeLane < 4) {
          setLaneGreen(activeLane);
          Serial.print("GREEN: ");
          Serial.println(lanes[activeLane].name);
          currentState = STATE_GREEN;
          stateStartTime = now;
        } else {
          currentState = STATE_IDLE;
        }
      }
      break;

    case STATE_GREEN:
      // Green phase — wait for duration
      if (elapsed >= greenDuration) {
        // Transition to yellow
        setLaneYellow(activeLane);
        Serial.print("YELLOW: ");
        Serial.println(lanes[activeLane].name);
        currentState = STATE_YELLOW;
        stateStartTime = now;
      }
      break;

    case STATE_YELLOW:
      // Yellow phase — wait for YELLOW_TIME
      if (elapsed >= YELLOW_TIME) {
        // Turn all red
        setAllRed();
        currentState = STATE_POST_RED;
        stateStartTime = now;
      }
      break;

    case STATE_POST_RED:
      // Brief all-red after yellow
      if (elapsed >= ALL_RED_TIME) {
        if (sequenceMode) {
          // Move to next lane in sequence
          currentState = STATE_SEQUENCE_NEXT;
          stateStartTime = now;
        } else {
          // Single priority mode — done
          Serial.println("CYCLE_DONE");
          Serial.println("READY");
          currentState = STATE_IDLE;
        }
      }
      break;

    case STATE_SEQUENCE_NEXT:
      // Check if there are more lanes in the sequence
      sequenceIndex++;
      if (sequenceIndex < 4) {
        // Skip lanes with 0 time
        if (laneTimes[sequenceIndex] <= 0) {
          currentState = STATE_SEQUENCE_NEXT;
        } else {
          activeLane = sequenceIndex;
          greenDuration = laneTimes[sequenceIndex];
          setAllRed();
          currentState = STATE_ALL_RED;
          stateStartTime = millis();
        }
      } else {
        // All lanes done
        sequenceMode = false;
        Serial.println("SEQUENCE_DONE");
        Serial.println("READY");
        currentState = STATE_IDLE;
      }
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SERIAL COMMAND PARSER
// ═══════════════════════════════════════════════════════════════════

void checkSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  Serial.print("CMD: ");
  Serial.println(cmd);

  // Check if it's a single lane command: "N", "E", "S", or "W"
  if (cmd.length() == 1) {
    char lane = cmd.charAt(0);
    int laneIdx = getLaneIndex(lane);

    if (laneIdx >= 0) {
      // Single lane priority mode
      sequenceMode = false;
      activeLane = laneIdx;
      greenDuration = DEFAULT_GREEN_TIME;

      Serial.print("PRIORITY: ");
      Serial.println(lane);

      // Start the cycle
      setAllRed();
      currentState = STATE_ALL_RED;
      stateStartTime = millis();
    } else {
      Serial.println("ERR: Invalid lane");
    }
  }
  // Check for timing format: "N:10,E:15,S:8,W:12"
  else if (cmd.indexOf(':') > 0) {
    if (parseTimings(cmd)) {
      sequenceMode = true;
      sequenceIndex = 0;

      // Find first non-zero lane
      while (sequenceIndex < 4 && laneTimes[sequenceIndex] <= 0) {
        sequenceIndex++;
      }

      if (sequenceIndex < 4) {
        Serial.println("SEQUENCE_START");
        setAllRed();
        currentState = STATE_ALL_RED;
        stateStartTime = millis();
      } else {
        Serial.println("ERR: All timings zero");
        currentState = STATE_IDLE;
      }
    } else {
      Serial.println("ERR: Bad timing format");
    }
  }
  // Check for special command: "STOP"
  else if (cmd == "STOP") {
    setAllRed();
    currentState = STATE_IDLE;
    Serial.println("STOPPED");
    Serial.println("READY");
  }
  // Unknown command
  else {
    Serial.println("ERR: Unknown command");
  }
}

bool parseTimings(String cmd) {
  // Expected format: "N:10,E:15,S:8,W:12"
  // Reset all timings
  for (int i = 0; i < 4; i++) laneTimes[i] = 0;

  int start = 0;
  while (start < (int)cmd.length()) {
    int commaPos = cmd.indexOf(',', start);
    String part;
    if (commaPos < 0) {
      part = cmd.substring(start);
      start = cmd.length();
    } else {
      part = cmd.substring(start, commaPos);
      start = commaPos + 1;
    }

    part.trim();
    int colonPos = part.indexOf(':');
    if (colonPos < 0) return false;

    char lane = part.charAt(0);
    int seconds = part.substring(colonPos + 1).toInt();
    int laneIdx = getLaneIndex(lane);

    if (laneIdx < 0) return false;

    // Convert seconds to milliseconds, with bounds
    unsigned long ms = (unsigned long)seconds * 1000;
    if (ms < MIN_GREEN_TIME) ms = MIN_GREEN_TIME;
    if (ms > MAX_GREEN_TIME) ms = MAX_GREEN_TIME;
    laneTimes[laneIdx] = ms;
  }

  return true;
}

int getLaneIndex(char lane) {
  switch (lane) {
    case 'N': case 'n': return 0;
    case 'E': case 'e': return 1;
    case 'S': case 's': return 2;
    case 'W': case 'w': return 3;
    default: return -1;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SIGNAL CONTROL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

void setAllRed() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(lanes[i].pinR, HIGH);
    digitalWrite(lanes[i].pinY, LOW);
    digitalWrite(lanes[i].pinG, LOW);
  }
}

void setLaneGreen(int idx) {
  // Set all to red first
  setAllRed();
  // Then set target lane to green
  digitalWrite(lanes[idx].pinR, LOW);
  digitalWrite(lanes[idx].pinG, HIGH);
}

void setLaneYellow(int idx) {
  // Turn off green, turn on yellow for the active lane
  digitalWrite(lanes[idx].pinG, LOW);
  digitalWrite(lanes[idx].pinY, HIGH);
}

void setAllYellow() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(lanes[i].pinR, LOW);
    digitalWrite(lanes[i].pinY, HIGH);
    digitalWrite(lanes[i].pinG, LOW);
  }
}
