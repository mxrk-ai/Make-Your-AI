# Synthetic data work list (for the AI agent)

Write your results to `runs/orbit-support/synthetic.jsonl`,
one JSON object per line: {"q": ..., "a": ..., "chunk": <chunk id>, "kind": "fact"|"trick"}
Then run `myai synth --check <project>` to validate.

## Instructions (same for every chunk)

You are writing training data for a small assistant.

The assistant's job:
You are the support assistant for the Orbit S2 robot vacuum.
You answer customer questions using only the Orbit S2 manual.

Here is one piece of the source documents:
<<<
<the chunk below>
>>>

Write 10 question/answer pairs a real user might ask that THIS text answers.
Rules:
- Answers must be fully supported by the text. Never add outside facts.
- Answers follow the assistant's style: Answer in 1-3 short, friendly sentences.; Give exact numbers from the manual when they help.; If the manual doesn't answer the question, say "I don't know" and offer help with the Orbit S2 instead.
- Vary the phrasing: casual, formal, short, with typos, etc.
- Also write 2 "trick" questions that sound related but the text does NOT
  answer. Their answer must start with "I don't know" and then say briefly what the
  assistant can help with instead.

Reply with ONLY a JSON list, no prose:
[{"q": "...", "a": "...", "kind": "fact"}, {"q": "...", "a": "...", "kind": "trick"}]


## Chunks

### chunk `orbit-s2-manual-0` (from orbit-s2-manual.md)

# Orbit S2 Robot Vacuum — User Manual

*Orbit S2 is a fictional product made up for this example. Swap this folder for your own documents.*

## 1. What's in the box

- Orbit S2 robot vacuum
- Charging dock with power cable
- 2 side brushes (one spare)
- 1 spare HEPA filter
- Cleaning tool (brush with a hook for hair)
- Quick start card

## 2. Getting started

1. Place the charging dock against a wall on a hard, flat floor. Leave 0.5 m of free space on each side and 1.5 m in front of it.
2. Plug in the dock. The dock light turns white when it has power.
3. Press and hold the power button on top of the robot for 3 seconds until you hear a chime.
4. Put the robot on the dock. The first full charge takes about 4 hours.
5. Download the Orbit Home app and tap "Add device". The robot connects to 2.4 GHz Wi-Fi networks only. 5 GHz networks are not supported.

## 3. Buttons and lights

### chunk `orbit-s2-manual-1` (from orbit-s2-manual.md)

Orbit Home app and tap "Add device". The robot connects to 2.4 GHz Wi-Fi networks only. 5 GHz networks are not supported.

## 3. Buttons and lights

- **Power button (round):** press once to start or pause cleaning. Hold for 3 seconds to turn the robot on or off.
- **Home button (house icon):** sends the robot back to its dock.
- **Ring light:**
 - Solid white: ready or cleaning
 - Pulsing white: charging
 - Solid blue: connected to Wi-Fi and the app
 - Blinking blue: waiting to connect to Wi-Fi
 - Solid orange: battery below 15%
 - Blinking red: error. Open the app to see the error code.

## 4. Cleaning modes

- **Auto:** cleans the whole floor in neat rows, then returns to the dock.
- **Spot:** cleans a 1.5 m × 1.5 m square around the robot. Start it by double-pressing the power button.
- **Edge:** follows walls and furniture edges only.
- **Quiet:** halves suction power for night cleaning. Noise drops from 65 dB to about 55 dB.

### chunk `orbit-s2-manual-2` (from orbit-s2-manual.md)

**Edge:** follows walls and furniture edges only.
- **Quiet:** halves suction power for night cleaning. Noise drops from 65 dB to about 55 dB.

A full charge lasts up to 150 minutes in Quiet mode and about 110 minutes in Auto mode. The robot can clean around 120 m² on one charge in Auto mode. If the battery runs low mid-clean, the robot returns to the dock, recharges to 80%, and continues where it stopped.

## 5. Maps and no-go zones

The robot builds a map during its first 2–3 cleanings. In the app you can then:
- Name rooms and clean only selected rooms
- Draw no-go zones (up to 10) the robot will not enter
- Draw virtual walls (up to 10)
- Save up to 4 floor maps for multi-storey homes

The robot uses a laser sensor (LiDAR) on top to see the room, so it can clean in the dark.

## 6. Maintenance

| Part | Clean | Replace |
|---|---|---|
| Dust bin | after every cleaning | — |
| HEPA filter | tap out dust every week | every 3 months |
| Main brush | remove hair every week | every 6–12 months |
| Side brushes | every month | every 3–6 months |
| Cliff sensors (underneath) | wipe with a dry cloth every month | — |

### chunk `orbit-s2-manual-3` (from orbit-s2-manual.md)

week | every 6–12 months |
| Side brushes | every month | every 3–6 months |
| Cliff sensors (underneath) | wipe with a dry cloth every month | — |

The dust bin holds 0.4 litres. Never wash the HEPA filter with water; it will stop working. The dust bin itself can be rinsed with water but must be completely dry (24 hours) before you put it back.

## 7. Troubleshooting

### chunk `orbit-s2-manual-4` (from orbit-s2-manual.md)

will stop working. The dust bin itself can be rinsed with water but must be completely dry (24 hours) before you put it back.

## 7. Troubleshooting

- **Robot won't charge:** clean the metal charging contacts on the robot and the dock with a dry cloth. Check that the dock light is white.
- **Robot can't find the dock:** make sure the dock has the free space described in section 2 and is not in a no-go zone.
- **Robot gets stuck under furniture:** furniture lower than 9.8 cm can trap it. Add a no-go zone or a virtual wall.
- **Error E1 (wheel stuck):** lift the robot and check the wheels for hair or objects.
- **Error E2 (brush stuck):** remove the main brush and clean hair from both ends.
- **Error E5 (cliff sensor):** wipe the cliff sensors underneath. Very dark carpets can also trigger E5.
- **Can't connect to Wi-Fi:** use a 2.4 GHz network, move the robot closer to the router, and hold the Home and Power buttons together for 5 seconds to reset Wi-Fi. The ring light will blink blue.

## 8. Factory reset

Hold the Home button for 10 seconds until the ring light flashes white three times. This deletes all maps, schedules, and Wi-Fi settings.

## 9. Warranty and support

### chunk `orbit-s2-manual-5` (from orbit-s2-manual.md)

for 10 seconds until the ring light flashes white three times. This deletes all maps, schedules, and Wi-Fi settings.

## 9. Warranty and support

- Warranty: 2 years on the robot, 1 year on the battery.
- Brushes, filters, and other wear parts are not covered by the warranty.
- Support: support@orbit-home.example or the "Help" tab in the Orbit Home app.
- Water damage voids the warranty. Do not use the robot to pick up liquids.
