#!/bin/bash -x

# Install build tools
pip2 install --user scons pyscons pipenv

# Create the pipenv environment
pipenv install

# Run scons to build all the figures
scons paper/figures/

# Build the paper
cd paper
xelatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex

# Force push the paper to GitHub
cd $TRAVIS_BUILD_DIR
git checkout --orphan $TRAVIS_BRANCH-pdf
git rm -rf .
git add -f paper/paper.pdf
git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
