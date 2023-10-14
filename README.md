# polish-nlm
Development of natural language models for the Polish language.

## Development

Development is done in a docker container. To get started, copy the `.env-template`
file and populate the values in the `.env` file:

```
cp .env-template .env
```

Run the command below to start the container:

```
docker compose up -d
```

Attach to the running container to then run further commands:

```
docker compose exec -it dev /bin/bash
```
