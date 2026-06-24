This repo contains the code for the legend monitoring dashboard.
The paths to the relevant data are contained in the `dashboard-config.yaml` file at the the project root.
The dashboard uses the panel framework : https://panel.holoviz.org/, and
consists of a series of classes which inherit from each other to share underlying parameters.
It runs on the spin system at NERSC inside a lightweight docker image:
https://docs.nersc.gov/services/spin/.
It should be servable to multiple people, giving them independent dashboards they can
interact with. It should be fast and responsive for users with minimal spin up time.
All code is formatted using pre-commit with `pre-commit run -a`.
