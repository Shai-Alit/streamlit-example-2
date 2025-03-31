# streamlit-example-2

This example contains everything needed to run a model inside a Docker container
src - model training code
    - scoring code for testing
app - requirements for python environment
    - score code for deployment
    - settings for deployed environment
    - application front end code
models - serialized models for deployed application
Dockerfile - defines docker container
action.yml - defines GitHub action for automatic deployment 
environment - defines environment for docker
run.sh - runs container
