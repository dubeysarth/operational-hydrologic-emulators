import os
import torch
import pandas as pd

class LumpedDataLoader:
    def __init__(self, PATH_DATASET):
        self.PATH_DATASET = PATH_DATASET

        timestamps = torch.load(os.path.join(PATH_DATASET, 'timestamps.pt'))
        timestamps = pd.DataFrame(index=timestamps)
        timestamps.index.name = 'timestamp'
        self.timestamps = timestamps

        self.static = torch.load(os.path.join(PATH_DATASET, 'X_static.pt'))

        self.dynamic = {
            'ERA5': torch.load(os.path.join(PATH_DATASET, 'X_dynamic_ERA5.pt')),
            'GPM': torch.load(os.path.join(PATH_DATASET, 'X_dynamic_GPM.pt')),
            'HRES': torch.load(os.path.join(PATH_DATASET, 'X_dynamic_HRES.pt')),
            'encodings': torch.load(os.path.join(PATH_DATASET, 'encodings.pt')),
            'Prcp': torch.load(os.path.join(PATH_DATASET, 'Prcp.pt')).transpose(1, 0),
            'PET': torch.load(os.path.join(PATH_DATASET, 'PET.pt')).transpose(1, 0)
        }

        self.sim = torch.load(os.path.join(PATH_DATASET, 'y_sim.pt'))

    def get_shapes(self):
        print("dynamic:")
        for key, value in self.dynamic.items():
            print(f"\t{key}: {value.shape}")
        print(f"static: {self.static.shape}")
        print(f"sim: {self.sim.shape}")

    def _set_time_period(self, start_date='2016-10-01', end_date='2019-09-30', lag=365, lead=10, verbose=True):
        self.start_idx = self.timestamps.index.get_loc(start_date)
        self.end_idx = self.timestamps.index.get_loc(end_date)
        self.lag = lag
        self.lead = lead
        max_idx = (self.end_idx - self.start_idx + 1) - (self.lag + self.lead)
        self.available_indices = list(range(max_idx))
        if verbose:
            print(f"Start index: {self.start_idx}, End index: {self.end_idx}, Max sample index: {max_idx}, Lag: {self.lag}, Lead: {self.lead}")

    def get_sample_window(self, idx, verbose=False):
        lag_start = self.start_idx + idx
        lag_end = lag_start + self.lag - 1
        lead_start = lag_end + 1
        lead_end = lead_start + self.lead - 1

        if verbose:
            print(f"lag start: {self.timestamps.index[lag_start].date()} (at index {lag_start})")
            print(f"lag end: {self.timestamps.index[lag_end].date()} (at index {lag_end})")
            print(f"lead start: {self.timestamps.index[lead_start].date()} (at index {lead_start})")
            print(f"lead end: {self.timestamps.index[lead_end].date()} (at index {lead_end})")

            print(f"lag start-end: {lag_end - lag_start + 1} days")
            print(f"lead start-end: {lead_end - lead_start + 1} days")
        
        return lag_start, lag_end, lead_start, lead_end
    
    def get_sample(self, idx, verbose=False):
        lag_start, lag_end, lead_start, lead_end = self.get_sample_window(idx, verbose=verbose)

        # Need 0/1 tensors to show when which data is available for model.    
        lag_era5_available = torch.ones(self.lag, dtype=torch.bool)
        lag_era5_available[-5:] = False # Set last 5 days to zero
        
        lag_gpm_final_available = torch.ones(self.lag, dtype=torch.bool)
        lag_gpm_final_available[-90:] = False # Set last 90 days to zero
        
        lag_gpm_late_available = torch.ones(self.lag, dtype=torch.bool)

        lead_hres_available = torch.ones(self.lead, dtype=torch.bool)
        
        # Tensors
        lag_encodings = self.dynamic['encodings'][lag_start:lag_end + 1]
        lead_encodings = self.dynamic['encodings'][lead_start:lead_end + 1]

        X_dynamic_ERA5 = self.dynamic['ERA5'][lag_start:lag_end + 1]
        X_dynamic_ERA5 = torch.cat([X_dynamic_ERA5, lag_encodings], dim=-1)

        X_dynamic_GPM_Final = self.dynamic['GPM'][lag_start:lag_end + 1, :, 2:3]
        X_dynamic_GPM_Final = torch.cat([X_dynamic_GPM_Final, lag_encodings], dim=-1)

        X_dynamic_GPM_Late = self.dynamic['GPM'][lag_start:lag_end + 1, :, 1:2]
        X_dynamic_GPM_Late = torch.cat([X_dynamic_GPM_Late, lag_encodings], dim=-1)

        X_dynamic_HRES = self.dynamic['HRES'][lag_end, :, :self.lead, :].permute(1, 0, 2)
        X_dynamic_HRES = torch.cat([X_dynamic_HRES, lead_encodings], dim=-1)

        X_dynamic_ERA5_lead = self.dynamic['ERA5'][lead_start:lead_end + 1]
        X_dynamic_ERA5_lead = torch.cat([X_dynamic_ERA5_lead, lead_encodings], dim=-1)

        X_dynamic_GPM_Final_Lead = self.dynamic['GPM'][lead_start:lead_end + 1, :, 2:3]
        X_dynamic_GPM_Final_Lead = torch.cat([X_dynamic_GPM_Final_Lead, lead_encodings], dim=-1)
        
        X_static = self.static

        y_sim = self.sim[lead_start:lead_end + 1]

        Prcp_lag = self.dynamic['Prcp'][lag_start:lag_end + 1]
        Prcp_lead = self.dynamic['Prcp'][lead_start:lead_end + 1]
        PET_lag = self.dynamic['PET'][lag_start:lag_end + 1]
        PET_lead = self.dynamic['PET'][lead_start:lead_end + 1]

        sample = {
            'X_dynamic_ERA5': X_dynamic_ERA5,
            'X_dynamic_GPM_Final': X_dynamic_GPM_Final,
            'X_dynamic_GPM_Late': X_dynamic_GPM_Late,
            'X_dynamic_HRES': X_dynamic_HRES,
            'X_dynamic_ERA5_lead': X_dynamic_ERA5_lead,
            'X_dynamic_GPM_Final_Lead': X_dynamic_GPM_Final_Lead,
            'X_static': X_static,
            'y_sim': y_sim,
            'lag_encodings': lag_encodings,
            'lead_encodings': lead_encodings,
            'Prcp_lag': Prcp_lag,
            'Prcp_lead': Prcp_lead,
            'PET_lag': PET_lag,
            'PET_lead': PET_lead
        }

        available = {
            'lag_era5': lag_era5_available,
            'lag_gpm_final': lag_gpm_final_available,
            'lag_gpm_late': lag_gpm_late_available,
            'lead_hres': lead_hres_available,
        }

        if verbose:
            for key, value in sample.items():
                print(f"{key}: {value.shape}")

            for key, value in available.items():
                print(f"{key}: {value.shape}")
        
        return sample, available