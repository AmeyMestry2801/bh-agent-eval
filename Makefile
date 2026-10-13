# Behavioral-Health Intake and Risk Agent: evaluation harness. ALL DATA IS SYNTHETIC.
# Run `make all` in a FRESH CLONE. Do not run `make clean` in a working copy that has
# a test-set result you want to keep: it removes results/TEST_RUN.lock (test-once guard).
PY ?= python
S := results/.stamps

.PHONY: all test llm clean
all: $(S)/bi test

data/raw/conversations.csv:
	$(PY) -m src.generate_data

$(S)/validate: data/raw/conversations.csv
	$(PY) -m src.validate
	mkdir -p $(S) && touch $@

$(S)/eda: $(S)/validate
	$(PY) notebooks/_py2nb.py notebooks/01_eda.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	mkdir -p $(S) && touch $@

$(S)/split: $(S)/validate
	$(PY) -m src.split
	mkdir -p $(S) && touch $@

$(S)/models: $(S)/split
	$(PY) -m src.models
	mkdir -p $(S) && touch $@

$(S)/promote: $(S)/models
	$(PY) -m src.promote
	mkdir -p $(S) && touch $@

$(S)/calibrate: $(S)/promote
	$(PY) -m src.calibrate
	mkdir -p $(S) && touch $@

$(S)/final_test: $(S)/calibrate
	$(PY) -m src.final_test
	mkdir -p $(S) && touch $@

$(S)/slices: $(S)/final_test
	$(PY) -m src.slices
	mkdir -p $(S) && touch $@

$(S)/errors: $(S)/slices
	$(PY) -m src.error_analysis
	mkdir -p $(S) && touch $@

$(S)/drift: $(S)/final_test
	$(PY) -m src.drift
	mkdir -p $(S) && touch $@

$(S)/audit: $(S)/drift
	$(PY) -m src.audit
	mkdir -p $(S) && touch $@

$(S)/bi: $(S)/audit $(S)/errors $(S)/eda
	$(PY) -m src.export_bi
	mkdir -p $(S) && touch $@

test:
	$(PY) -m pytest -q

# Optional: needs GROQ_API_KEY in .env. Not part of `make all`.
llm:
	$(PY) -m src.llm_agent --run

clean:
	rm -rf data/raw data/processed results/.stamps reports/figures/*.png bi/*.csv bi/*.xlsx
	find results -mindepth 1 ! -name RESULTS.md ! -name resume_facts.json -delete
