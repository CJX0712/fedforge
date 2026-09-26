PY = C:/Users/Administrator/.workbuddy/binaries/python/envs/fedforge/Scripts/python.exe

.PHONY: install test demo train hpo clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q -W ignore::UserWarning

demo:
	$(PY) examples/run_demo.py

train:
	$(PY) -m fedforge.cli train --strategy fedavg --partition dirichlet --alpha 0.1

hpo:
	$(PY) -m fedforge.cli hpo --trials 10

clean:
	rm -f benchmark.json
	find . -type d -name __pycache__ -exec rm -rf {} +
