FROM continuumio/miniconda3
COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"
RUN conda config --add channels conda-forge
RUN conda install --file /tmp/requirements.txt
EXPOSE 8050
ENTRYPOINT [ "python3" ]
CMD [ "dash_app.py" ]
