#!/bin/zsh
coverage run --source=src -m pytest
coverage report
