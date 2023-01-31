# geopyv
Python/C++ based PIV/DIC package.

# Development instructions

## Setup

Use pipenv in root folder of project:
- pip install pipenv
- pipenv shell
[- cd external
- git submodule update --init
- cd ..]
- pipenv install .
- pipenv uninstall geopyv

## Docs using sphinx
Using sphinx and 'Read the Docs' theme:

pip install sphinx spinx-rtd-theme

In the /docs folder:
- make clean html

Open index.html in build to view the contents.

Use numpy docstrings formatting (via napoleon extension to sphinx, which is installed in
conf.py).

## Tests using pytest
Use pytests to execute tests in the /tests folder: pipenv install pytest

- All files in tests folder
- All files named 'tests_*.py' where * is the module, function or class name
- See https://www.youtube.com/watch?v=etosV2IWBF0 for an introduction

## Version control

### Basics
- git status - checks status of local repo
- git add . - adds all files to staging area
- git commit -m "Message..." - commits staged files with the message outlined
- git push - Pushes commit to GitHub

### To fork and write a feature
Follow this guide: http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/

## Code style
Please check code style before commits...

### Formatting
Use black: https://black.readthedocs.io/en/stable/the_black_code_style.html

If using pipenv install via: pipenv install black --pre

Don't forget the --pre or it will cause problems!

### PEP8 compliance
Check PEP8 compliance using flake8: pipenv install flake8
