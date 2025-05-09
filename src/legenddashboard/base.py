import time
import logging
from pathlib import Path
from datetime import datetime, date
from dbetto import Props
import numpy as np
import bisect
import datetime as dtt
import param

from legenddashboard.util import gen_run_dict

from bokeh.io import output_notebook
from bokeh.resources import INLINE

log = logging.getLogger(__name__)

class Monitoring(param.Parameterized):
    """
    Base class for monitoring dashboards.
    """
    base_path = param.String("")
    prod_config = param.Dict({})
    tier_dict = param.Dict({})
    period = param.Selector(default="p00", objects=[f"p{i:02}" for i in range(100)], allow_refs=True, nested_refs=True)
    run = param.Selector(default="r000", objects=[f"r{i:03}" for i in range(100)], allow_refs=True, nested_refs=True)
    run_dict = param.Dict({}, allow_refs=True, nested_refs=True)
    periods = param.Dict({}, allow_refs=True, nested_refs=True)
    
    date_range = param.DateRange(
            default=(
                datetime.now() - dtt.timedelta(minutes=10),
                datetime.now() + dtt.timedelta(minutes=10),
            ),
            bounds=(
                datetime(2000,1,1,0,0,0),
                datetime(2100,1,1,0,0,0),
            ), allow_refs=True, nested_refs=True
        )
    

    def __init__(self, base_path,  notebook=False, **params):
        if notebook is True:
            output_notebook(INLINE)
        self.cached_plots = {}
        self.base_path = base_path

        self.startup_bool = True
        super().__init__(**params)

        self.tier_dict = {
            "raw":"raw",
            "tcm":"tcm",
            "dsp":"dsp",
            "hit":"hit",
            "evt":"evt",
        }

        if "ref-v" in str(self.base_path):
            self.tier_dict["dsp"] = "psp"
            self.tier_dict["hit"] = "pht"
            self.tier_dict["evt"] = "pet"
            
        prod_config = Path(self.base_path) / "dataflow-config.yaml"
        self.prod_config = Props.read_from(prod_config, subst_pathvar=True)
        if self.period == "p00":
            self.periods = gen_run_dict(self.base_path)
            print("updating")
            self.param["period"].objects = list(self.periods)
            self.period=list(self.periods)[0]
            self._get_period_data()

    @param.depends("period", watch=True)
    def _get_period_data(self):
        if self.startup_bool:
            log.debug("Startup procedure, skip _get_period_data")
            self.startup_bool = False
        else:
            self.run_dict = self.periods[self.period]

            self.param["run"].objects = list(self.run_dict)
            if self.run == list(self.run_dict)[-1]:
                self.run = next(iter(self.run_dict))
            else:
                self.run = list(self.run_dict)[-1]

            start_period = sorted(self.periods)[0]
            start_run = sorted(self.periods[start_period])[0]
            end_period = sorted(self.periods)[-1]
            end_run = sorted(self.periods[end_period])[-1]

            self.param["date_range"].bounds = (
                datetime.strptime(
                    self.periods[start_period][start_run]["timestamp"], "%Y%m%dT%H%M%SZ"
                )
                - dtt.timedelta(minutes=100),
                datetime.strptime(
                    self.periods[end_period][end_run]["timestamp"], "%Y%m%dT%H%M%SZ"
                )
                + dtt.timedelta(minutes=110),
            )
            self.date_range = (
                datetime.strptime(
                    self.periods[start_period][start_run]["timestamp"], "%Y%m%dT%H%M%SZ"
                )
                - dtt.timedelta(minutes=100),
                datetime.strptime(
                    self.periods[end_period][end_run]["timestamp"], "%Y%m%dT%H%M%SZ"
                )
                + dtt.timedelta(minutes=110),
            )

    @param.depends("date_range", watch=True)
    def _get_run_dict(self):
        start_time = time.time()
        valid_from = [
            datetime.timestamp(
                datetime.strptime(self.run_dict[entry]["timestamp"], "%Y%m%dT%H%M%SZ")
            )
            for entry in self.run_dict
        ]
        if isinstance(self.date_range[0], date):
            low_range = datetime.timestamp(
                datetime.combine(self.date_range[0], datetime.min.time())
            )
        else:
            low_range = datetime.timestamp(self.date_range[0])
        if isinstance(self.date_range[0], date):
            high_range = datetime.timestamp(
                datetime.combine(self.date_range[1], datetime.max.time())
            )
        else:
            high_range = datetime.timestamp(self.date_range[1])
        pos1 = bisect.bisect_right(valid_from, low_range)
        pos2 = bisect.bisect_left(valid_from, high_range)
        pos1 = max(pos1, 0)
        pos2 = min(len(self.run_dict), pos2)
        valid_idxs = np.arange(pos1, pos2, 1)
        valid_keys = np.array(list(self.run_dict))[valid_idxs]
        out_dict = {key: self.run_dict[key] for key in valid_keys}
        log.debug("Time to get run dict:", extra={"time": time.time() - start_time})
        return out_dict
