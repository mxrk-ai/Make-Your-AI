# Design first, then train: plant watering

`data.csv` has 400 readings from a (made-up) plant sensor: soil moisture (%), temperature (°C),
hours since the last watering, and whether the plant needed water (`label` 1) or not (0).

`network.json` is a hand-written **first guess** ("water below 30% moisture"). It ignores temperature
on purpose, because the right temperature effect wasn't obvious from the rows. Then training starts
from those weights:

```bash
myai scratch test  examples/scratch-plant/network.json --csv examples/scratch-plant/data.csv
myai scratch train examples/scratch-plant/network.json --csv examples/scratch-plant/data.csv
```

Scores count only the 30% of rows that training never sees.

How the data was made, so you can check the results: a plant needs water when
`moisture < 30 + (temperature - 20)`, `hours_since_water` doesn't matter, and 3% of labels
were flipped at random as sensor noise. So about 97% is the best any network can score.
