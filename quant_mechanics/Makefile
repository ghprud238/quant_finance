.PHONY: test demo clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

demo:
	PYTHONPATH=src python3 scripts/demo_modules_51_52.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
