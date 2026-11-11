.PHONY: install test doctor

install:
	./install.sh

test:
	.venv/bin/pytest -q

doctor:
	.venv/bin/myai doctor
