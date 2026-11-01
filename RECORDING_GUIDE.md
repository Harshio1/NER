# Recording guide: your 10 + 3 sentences

Read each sentence **naturally, in your own voice**, one file per sentence. Pause for a second before and after.
Record on a phone voice recorder or Audacity. WAV, OGG, MP3 and M4A all work with Whisper.

## Training set: save into `audio/train/`

| File | Sentence |
|---|---|
| sentence_01.ogg | Sundar Pichai is the chief executive of Google. |
| sentence_02.ogg | Infosys opened a new office in Bangalore. |
| sentence_03.ogg | I visited Chennai with my friend Sreenedhi last summer. |
| sentence_04.ogg | Abdul Kalam was born in Rameshwaram. |
| sentence_05.ogg | The Indian Space Research Organization launched a rocket from Sriharikota. |
| sentence_06.ogg | Virat Kohli played for Royal Challengers Bangalore. |
| sentence_07.ogg | Anna University is located in Kandy near Chennai. |
| sentence_08.ogg | The Reserve Bank of India is headquartered in Mumbai. |
| sentence_09.ogg | Kalpana Chawla worked for NASA. |
| sentence_10.ogg | Tesla opened a showroom in Delhi. |

## Test set (new recordings): save into `audio/test/`

| File | Sentence |
|---|---|
| test_01.ogg | Narendra Modi visited Madurai and met officials from Wipro. |
| test_02.ogg | Mahendra Singh Dhoni played for Chennai Super Kings. |
| test_03.ogg | Smriti Mandhana joined Deloitte in Vishakhapatnam. |

Feel free to replace any sentence with your own (names of your friends, your college, your town). If you do, edit the
`marked` column of `data/sentences.tsv` using `[entity text|PER]`, `[...|ORG]`, `[...|LOC]`, and keep the test sentences
free of entities that appear in training.

## After recording
1. `python src/transcribe.py --audio audio/train` : speech to text, with WER against your script.
2. If Whisper heard something different from what you intended, correct the `marked` line so tokens match the transcript.
3. `python src/annotate.py build --in data/sentences.tsv --prefix data/annotated_train`.
4. Re-run `cd src && python run_experiments.py && python make_report.py` to regenerate results and the website data.

## Google Drive submission checklist
- [ ] Folder 1 **"Audio recordings"** containing the 10 training `.ogg` files (+ the test recordings): share → *Add the evaluator's email* (or "Anyone with the link: Viewer") → paste the link in the Google Form.
- [ ] Folder 2 **"Project"** containing this whole project folder (code, `data/`, `results/`, `report/`, `website/`).
- [ ] Open both links in a private window to confirm they work.
