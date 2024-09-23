FROM tensorflow/tensorflow:2.16.1-jupyter

WORKDIR /share

COPY . /share/

RUN python -m pip install -r requirements.txt

CMD jupyter nbconvert --execute --to html /script/minimal.ipynb