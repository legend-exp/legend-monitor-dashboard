
## Running the dashboard

It is recommended to use `uv` to run the dashboard as it makes the package management easier.
Clone the repo and update the `dashboard-config.yaml` with the desired paths then simply run:

` uv run dashboard dashboard-config.yaml -p 9009 `

to get the full dashboard. If running remotely you will just need to forward the relevant port.
It is also possible to disable certain pages if not needed/wanted e.g.: 

` uv run dashboard dashboard-config.yaml -p 9009 -d spm muon llama phy`

Alternatively can just run the individual componenets:

` uv run dashboard-cal dashboard-config.yaml -p 9009 `
` uv run dashboard-phy dashboard-config.yaml -p 9009 `
` uv run dashboard-llama dashboard-config.yaml -p 9009 `
` uv run dashboard-muon dashboard-config.yaml -p 9009 `
` uv run dashboard-spms dashboard-config.yaml -p 9009 `

## Developing

For developing it is recommending to install the package using the following commands:

`uv venv`
`source .venv/bin/activate`
`uv pip install -e .[dev]`

From there you can start a jupyter notebook. 
All the relevant classes have a `build_*_pane(s)` function to get the panel panes which can 
be displayed in a notebook. You should also pass the `notebook=True` option to make displaying 
work better.