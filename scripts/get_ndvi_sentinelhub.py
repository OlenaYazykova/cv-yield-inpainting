import pandas as pd
import geopandas as gpd
from rasterio.plot import show
from rasterio.io import MemoryFile
import rasterio
import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask

from sentinelhub import (
    SHConfig,
    Geometry,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubStatistical,
    SentinelHubRequest,
    bbox_to_dimensions,
    parse_time,
    BBox,
)

class Copernicus:

    def __init__(self, field_name, geometry, start_date = '2025-04-01T00:00:00Z', end_date = '2025-10-01T23:59:59Z'):

        # Credentials
        self.config = SHConfig()
        self.config.sh_client_id = os.getenv("SH_CLIENT_ID", "")
        self.config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "")

        load_dotenv()

        self.start_date = start_date
        self.end_date = end_date

        self.field_name = field_name
        self.crs = CRS.WGS84

        self.geometry = geometry

        self.geometrySH = Geometry(geometry=self.geometry, crs=CRS.WGS84)

        self.bbox = BBox(tuple(self.geometrySH.bbox), self.crs)

        self.ndvi_mean_df = []

        self.timestamp_list = []

        self.raster_list = []

        self.mask_list = []
        self.img_list = []

        self.file_path = None

        # Script for mean NDVI values and cloud/shadow pixel percentage
        self.evalscript_sentinel2_l2a = """
            //VERSION=3
            function setup() {
              return {
                input: [{
                  bands: [
                    "B04",
                    "B08",
                    "SCL",
                    "dataMask"
                  ]
                }],
                output: [
                  {
                    id: "data",
                    sampleType: "FLOAT32",
                    bands: 2
                  },
                  {
                    id: "dataMask",
                    bands: 1
                  }]
              };
            }
            
            function evaluatePixel(sample) {
              let ndvi = index(sample.B08, sample.B04);
              let scl = sample.SCL;
              let cloud_percent = 0;
              if (scl == 3) { // Cloud Shadows
                cloud_percent = 1;
              }  else if (scl == 7) { // Clouds low probability
                cloud_percent = 1;
              } else if (scl == 8) { // Clouds medium probability
                cloud_percent = 1;
              } else if (scl == 9) { // Clouds high probability
                cloud_percent = 1;
              } else if (scl == 10) { // Cirrus
                cloud_percent = 1;
              }
              return {
                    data: [ndvi, cloud_percent],
                    dataMask: [sample.dataMask]
                };
            }
            """
        
        # Script for retrieving single-band NDVI raster
        self.NDVI_S2_L2A = '''
            //VERSION=3
            
            function evaluatePixel(samples) {
                let ndvi = (samples.B08 - samples.B04)/(samples.B08 + samples.B04);
                return [ndvi];
            }
            
            function setup() {
              return {
                input: [{
                  bands: [
                    "B04",
                    "B08"
                  ]
                }],
                output: {
                  bands: 1,
                  sampleType: SampleType.FLOAT32
                }
              }
            }
            '''
        # Script for retrieving all-band L2A raster
        self.S2_L2A_ALL_BAND = '''
            //VERSION=3
            
            function evaluatePixel(samples) {
                let ndvi = (samples.B08 - samples.B04)/(samples.B08 + samples.B04);
                return [samples.B01, samples.B02, samples.B03, samples.B04, samples.B05, samples.B06, samples.B07, samples.B08, samples.B8A, samples.B09, samples.B11, samples.B12];
            }
            
            function setup() {
              return {
                input: [{
                  bands: [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                  ]
                }],
                output: {
                  bands: 12,
                  sampleType: SampleType.FLOAT32
                }
              }
            }
            '''

        self.evalscript_DEM = """
            //VERSION=3
            function setup() {
              return {
                input: ["DEM"],
                output:{
                  id: "default",
                  bands: 1,
                  sampleType: SampleType.FLOAT32
                }
              }
            }
            
            function evaluatePixel(sample) {
              return [sample.DEM]
            }
            """

    def _take_mean_data(self, geometry, evalscript, dataCollection, time_interval, resolution):
        '''Field statistics, mean NDVI values'''
        def stats_to_df(stats_data):
            """Transform Statistical API response into a pandas.DataFrame"""
            df_data = []
            for single_data in stats_data["data"]:
                df_entry = {}
                is_valid_entry = True
                df_entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()
                df_entry["interval_to"] = parse_time(single_data["interval"]["to"]).date()
                for output_name, output_data in single_data["outputs"].items():
                    for band_name, band_values in output_data["bands"].items():
                        band_stats = band_values["stats"]
                        if band_stats["sampleCount"] == band_stats["noDataCount"]:
                            is_valid_entry = False
                            break
                        for stat_name, value in band_stats.items():
                            col_name = f"{output_name}_{band_name}_{stat_name}"
                            if stat_name == "percentiles":
                                for perc, perc_val in value.items():
                                    perc_col_name = f"{col_name}_{perc}"
                                    df_entry[perc_col_name] = perc_val
                            else:
                                df_entry[col_name] = value
                if is_valid_entry:
                    df_data.append(df_entry)
            return pd.DataFrame(df_data)
    
        request = SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=evalscript,
                time_interval=time_interval,
                aggregation_interval='P1D',
                resolution=resolution,
            ),
            input_data=[
                SentinelHubStatistical.input_data(dataCollection, maxcc=0.8),
            ],
            geometry=geometry,
            config=self.config
        )
        response = request.get_data()
        ndvi_dfs = stats_to_df(response[0])
        ndvi_dfs['data_B0_mean'] = pd.to_numeric(ndvi_dfs['data_B0_mean'], errors='coerce')
        ndvi_dfs = ndvi_dfs.loc[ndvi_dfs['data_B0_mean'] <= 1]
        ndvi_dfs['interval_from'] = pd.to_datetime(ndvi_dfs['interval_from'])
        return ndvi_dfs

    def _take_raster_data(self, bbox, evalscript, dataCollection, date, resolution):
        '''Get rasters'''
        geom_size = bbox_to_dimensions(bbox, resolution=resolution)
        request_true_color = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=dataCollection,
                    time_interval=(date, date),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=geom_size,
            config=self.config,
        )
        
        data = request_true_color.get_data(decode_data=False)
        pus = float(data[0].headers['x-processingunits-spent'])
 
        return data, pus

    def take_mean_data(self,):
        # Fetch mean NDVI values over the field boundary for the given period
        self.ndvi_mean_df = self._take_mean_data(
                    geometry=self.geometrySH,
                    evalscript=self.evalscript_sentinel2_l2a,
                    dataCollection=DataCollection.SENTINEL2_L2A, 
                    time_interval=(self.start_date, self.end_date), 
                    resolution=(0.00009, 0.00009) # ~10 метрів
                )

        return self.ndvi_mean_df
        
    def cloud_filter(self, is_show=True):
        filtered_df = self.ndvi_mean_df[self.ndvi_mean_df['data_B1_mean'] == 0]

        # display(filtered_df[['interval_from','data_B0_mean', 'data_B1_mean']])
        
        values = [row['interval_from'] for index, row in filtered_df.iterrows()]
        
        self.timestamp_list = []
        for timestamp in values:
            date_string = timestamp.strftime('%Y-%m-%d')
            self.timestamp_list.append(date_string)

        print(f'cloud_filter for field {self.field_name}: Cloudy days dropped')

        if is_show == True:
            display(filtered_df[['interval_from','data_B0_mean', 'data_B1_mean']])

        return self.timestamp_list

    def get_rasters(self,):
        # Fetch NDVI pixels for the specified date
        
        # timestamp_list = timestamp_list[0:2]
        
        self.raster_list = []
        
        for idate in tqdm(self.timestamp_list, desc='Downloading'):
            raster, PUs = self._take_raster_data(
                bbox=self.geometrySH.bbox,
                evalscript=self.NDVI_S2_L2A,
                dataCollection=DataCollection.SENTINEL2_L2A,
                date=idate,
                resolution=10
            )
            self.raster_list.append(raster)

        return self.raster_list

    def _file_inmem_rio(self, bytes_content):
        '''Байти в проміжний файл rasterio'''
        with MemoryFile(bytes_content) as memfile:
            with memfile.open() as dataset:
                data_array = dataset.read()
        return data_array
    
    from rasterio.mask import mask
    
    def _file_inmem_rio_crop(self, bytes_content):
        '''Байти в проміжний файл rasterio'''
        with MemoryFile(bytes_content) as memfile:
            with memfile.open() as out_dest:

                data_array = out_dest.read()
                out_image, out_transform = mask(out_dest, [self.geometry], nodata=-9999, crop=True)
            memfile = []
            out_transform = []
            bytes_content = []
            
        return data_array, out_image
    
    def _crop_mask(self, image):
        ''' return image and field mask '''
        imgs, crop_image = self._file_inmem_rio_crop(image)  
        # Creating a mask for pixels with value -9999
        mask = (crop_image == -9999)    
        # Replace masked values with 0.0 and unmasked values with 1.0
        mask_crop = np.where(mask, 0.0, 1.0)
    
        return imgs, mask_crop

    def get_NDVI_TimeSeries(self,):
    # Creating a Matplotlib figure with subplots
    
        self.mask_list = []
        self.img_list = []

        self.get_rasters()
    
        for ii in range(len(self.raster_list)):
            
            img, mask = self._crop_mask(self.raster_list[ii][0].content)
            img = np.squeeze(img,0)
            mask = np.squeeze(mask,0)
            self.mask_list.append(mask)
            self.img_list.append(img)
    
        return self.img_list, self.mask_list, self.timestamp_list    

################# TIFF ###################################################################
    def _take_tiff_data(self, data_folder, filename, bbox, evalscript, dataCollection, date, resolution):
        '''Отримуємо растри'''
        geom_size = bbox_to_dimensions(bbox, resolution=resolution)
        request_true_color = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=dataCollection,
                    time_interval=(date, date),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=geom_size,
            config=self.config,
            data_folder=data_folder,
        )

        request_true_color.download_list[0].filename = f'{filename}'
        request_true_color.save_data()
        data = request_true_color.get_data(decode_data=False)

        return data 
    
    # Cut contour
    def save_TIFF(self, data_folder):
        print('save_TIFF', data_folder)
        os.makedirs(data_folder, exist_ok=True)

        self.raster_list = []

        for idate in tqdm(self.timestamp_list, desc='TIFF Downloading'):
            filename = f'{self.field_name}_{idate}.tiff'
            filepath = f'{data_folder}/{filename}'

            raster = self._take_tiff_data(
                data_folder=data_folder,
                filename=filename,
                bbox=self.geometrySH.bbox,
                evalscript=self.NDVI_S2_L2A,
                dataCollection=DataCollection.SENTINEL2_L2A,
                date=idate,
                resolution=10
            )

            data, out_image = self._file_inmem_rio_crop(raster[0].content)

            with MemoryFile(raster[0].content) as memfile:
                with memfile.open() as src:
                    meta = src.meta.copy()

            meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": src.transform,
                "nodata": -9999
            })

            with rasterio.open(filepath, "w", **meta) as dest:
                dest.write(out_image)

            self.file_path = filepath
            self.raster_list.append(out_image)

        return self.raster_list


    # def save_TIFF(self,)
    def _take_DEM_data(self, data_folder, filename, bbox, evalscript, dataCollection, date, resolution):
        '''Отримуємо растри'''
        geom_size = bbox_to_dimensions(bbox, resolution=resolution)
        request_true_color = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=dataCollection,
                    time_interval=("2020-06-12", "2020-06-13"),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=geom_size,
            config=self.config,
            data_folder=data_folder,
        )

        request_true_color.download_list[0].filename = f'{filename}'

        print('file name', request_true_color.download_list[0].filename)

        request_true_color.save_data()
        data = request_true_color.get_data(decode_data=False)

        return data


    def save_DEM(self, data_folder):
        # Fetch NDVI pixels for the specified date
        
        # timestamp_list = timestamp_list[0:2]
        
        raster = self._take_DEM_data(
            data_folder = f'{data_folder}', #/{idate}',
            filename = f'DEM_{self.field_name}.tiff',
            bbox=self.geometrySH.bbox.buffer(0.001),
            evalscript=self.evalscript_DEM,
            dataCollection=DataCollection.DEM,
            date='2024-01-01',
            resolution=30
        )
        # print('DEM:', type(raster), type(raster[0]))
############################################################
    
    def _plot_fields(self, is_show=True):

        if is_show == True:
            for ii in range(len(self.raster_list)):
                # print('mask:', np.max(crop_mask), np.min(crop_mask))
                raster = self.img_list[ii]
                print(f'NDVI stat: max={np.max(raster)}, min={np.min(raster)}')
        
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
                # Plotting array1 on the first subplot
                show(self.img_list[ii], ax=axs[0], cmap='viridis')
                axs[0].set_title(self.timestamp_list[ii])
                
                # Plotting array2 on the second subplot
                show(self.mask_list[ii]*self.img_list[ii], ax=axs[1], cmap='viridis')
                axs[1].set_title(self.timestamp_list[ii])
                
                # Display the plot
                plt.show()

    def _test_plot_ratser(self,):
    
        ii = 0
        
        raster = self.raster_list[ii][0].content
        print(f'NDVI stat: max={np.max(raster)}, min={np.min(raster)}')

        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    
        show(raster, ax=axs, cmap='viridis')
        axs.set_title(self.timestamp_list[ii])
        
        plt.show()
            
    def save_ndvi(self, svdir=''):
        # Saving both the array and the dates list into a .npz file
        np.savez(f'{svdir}{self.field_name}.npz', 
                 raster=self.img_list, 
                 mask=self.mask_list, 
                 time=self.timestamp_list, 
                 bbox=self.bbox,  
                 allow_pickle=True)

        print(f'ALL downloadings saved to {self.field_name}.npz')  





