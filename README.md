# streamlit-example-2

This example contains everything needed to run a model inside a Docker container </br>
src - model training code </br>
    - scoring code for testing </br>
app - requirements for python environment </br>
    - score code for deployment </br>
    - settings for deployed environment </br>
    - application front end code </br>
models - serialized models for deployed application </br>
Dockerfile - defines docker container </br>
action.yml - defines GitHub action for automatic deployment  </br>
environment - defines environment for docker </br>
run.sh - runs container </br>
