FROM continuumio/miniconda:latest

WORKDIR .
COPY . .

RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/yamlchem/bin:$PATH
RUN /bin/bash -c "source activate yamlchem"

CMD ["pylint", "yamlchem"]
CMD ["pylint", "tests"]
CMD ["pytest"]