SRC := src

style:
	pip install -r ./requirements/style.txt
	isort $(SRC)
	black $(SRC) --line-length 79
	flake8 $(SRC) --max-line-length 79

setup_gym:
	@echo install library
	pip install -r ./requirements/library.txt

	@echo setup for gym
	pip install -r ./requirements/gym.txt
	bash ./script/gym.sh

setup_brax:
	@echo install library
	pip install -r ./requirements/library.txt

	@echo setup for brax
	pip install -r ./requirements/brax.txt
	bash ./script/brax.sh

