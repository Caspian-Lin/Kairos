UV ?= uv
DEV_ENV_FILE ?= .env
SYSTEM_PYTHON ?= /usr/bin/python3

.PHONY: dev kairos napcat napcat-start napcat-stop launcher
dev:
	$(UV) run --env-file "$(DEV_ENV_FILE)" python scripts/dev.py

kairos:
	$(UV) run --env-file "$(DEV_ENV_FILE)" python scripts/dev.py kairos

napcat:
	$(SYSTEM_PYTHON) scripts/dev.py napcat

napcat-start:
	$(SYSTEM_PYTHON) scripts/dev.py napcat-start

napcat-stop:
	systemctl --user stop kairos-napcat.service

launcher:
	$(SYSTEM_PYTHON) scripts/launcher.py
