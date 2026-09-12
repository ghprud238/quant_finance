.PHONY: test demo clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

demo:
	PYTHONPATH=src python3 scripts/run_master_capstone.py

clean:
	rm -rf build dist *.egg-info output/*.png
