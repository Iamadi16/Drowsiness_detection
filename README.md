Drowsiness Detection
A real-time driver drowsiness detection system based on computer vision and facial analysis. The system monitors three main indicators of drowsiness:eye closure, 
yawning, and head-down position. Based on the detected events and their duration, a scoring system determines the driver's drowsiness level and activates appropriate warnings.

How It Works:
The system continuously analyzes the driver's face using a webcam and monitors three indicators:
Closed Eyes _ Yawning _ Head-Down Position
A specific time threshold is defined for each parameter. The driver's eyes must remain closed for 1.5 seconds, a yawn must last for 1 second, and the head must remain in
a downward position for 2 seconds for the corresponding event to be activated.

Since a single event, such as yawning, cannot definitively indicate that a driver is drowsy, the system uses a scoring mechanism to evaluate the driver's condition.
Drowsiness Scoring:
| Event       | Score |
| ----------- | ----: |
| Yawning     |    +1 |
| Closed Eyes |    +2 |
| Head Down   |    +2 |
Each event is counted only once during its occurrence to prevent the score from continuously increasing while the same event persists.

The total score determines the system's response:
| Score | System Response                                                                   |
| ----- | --------------------------------------------------------------------------------- |
| `< 3` | No warning                                                                        |
| `≥ 3` | Warning suggesting that the car windows be opened                                 |
| `≥ 6` | Audible warning informing the driver that they are drowsy and should take a break |
| `≥ 9` | Arduino-controlled buzzer is activated                                            |

Critical Eye-Closure Detection:
In addition to the scoring system, a critical eye-closure condition is implemented.
If the driver's eyes remain closed for more than 4 seconds, the buzzer is activated regardless of the current score. The buzzer operates for 5 seconds
and can be triggered again every 2 seconds until the driver opens their eyes.

Score Reset:
If no new drowsiness-related events are detected for 1 minute, the total score is reset to its initial state. This prevents events detected in the distant past from
continuing to influence the driver's current drowsiness assessment.

Hardware:
* Webcam
* Arduino
* Buzzer
