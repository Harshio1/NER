| Cell (bidirectional, 48 units) | Dev F1 | Test precision | Test recall | Test F1 | Test F1 (lower-case) |
|---|---|---|---|---|---|
| SimpleRNN | 94.0% | 66.7% | 25.0% | 36.4% | 0.0% |
| LSTM | 97.4% | 66.7% | 25.0% | 36.4% | 18.2% |
| GRU | 90.5% | 25.0% | 12.5% | 16.7% | 0.0% |

*Single training run (seed 0) of `src/train_keras.py --cell all`, executed locally with TensorFlow. With 8 test entities one correct
or wrong entity moves F1 by about 12 points, so differences between cells are within noise. Re-run `train_keras.py` to replace this
table: if `results/keras_results.json` exists, `make_report.py` uses it instead of this file.*
