.PHONY: test demo clean

test:
	python3 -m unittest discover -s tests -p "test_*.py"

demo:
	python3 scripts/run_all_demos.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
