
class ClipWorker(worker.AbstractWorker):
    def __init__(self, in_file, out_file, geojson, output_bounds=None):
        worker.AbstractWorker.__init__(self)

        self.in_file=in_file
        self.out_file=out_file
        self.output_bounds=output_bounds

        self.geojson=geojson

    def work(self):
        self.toggle_show_progress.emit(True)
        self.toggle_show_cancel.emit(True)

        json_file=GetTempFilename('.geojson')
        with open(json_file, 'w') as f:
            json.dump(self.geojson, f, separators=(',', ': '))

        gdal.UseExceptions()
        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format='GTiff',
            cutlineDSName=json_file,
            srcNodata=-32768,
            outputBounds=self.output_bounds,
            dstNodata=-32767,
            dstSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=[
                'NUM_THREADS=ALL_CPUs',
                'GDAL_CACHEMAX=500',
            ],
            creationOptions=[
                'COMPRESS=LZW',
                'NUM_THREADS=ALL_CPUs',
                'GDAL_NUM_THREADS=ALL_CPUs',
                'TILED=YES'
            ],
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None

    def progress_callback(self, fraction, message, data):
        if self.killed:
            return False
        else:
            self.progress.emit(100 * fraction)

            return True


class WarpWorker(worker.AbstractWorker):
    '''Used as a substitute for gdal translate given warp is multithreaded'''
    def __init__(self, in_file, out_file):
        worker.AbstractWorker.__init__(self)

        self.in_file = in_file
        self.out_file = out_file

    def work(self):
        self.toggle_show_progress.emit(True)
        self.toggle_show_cancel.emit(True)

        gdal.UseExceptions()

        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format='GTiff',
            srcNodata=-32768,
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=[
                'NUM_THREADS=ALL_CPUs',
                'GDAL_CACHEMAX=500',
            ],
            creationOptions=[
                'COMPRESS=LZW',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUs',
                'GDAL_NUM_THREADS=ALL_CPUs',
                'TILED=YES'
            ],
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback
        )
        if res:
            return True
        else:
            return None

    def progress_callback(self, fraction, message, data):
        if self.killed:
            return False
        else:
            self.progress.emit(100 * fraction)
            return True


class MaskWorker(worker.AbstractWorker):
    def __init__(self, out_file, geojson, model_file=None):
        worker.AbstractWorker.__init__(self)

        self.out_file=out_file
        self.geojson=geojson
        self.model_file=model_file

    def work(self):
        self.toggle_show_progress.emit(True)
        self.toggle_show_cancel.emit(True)

        json_file=GetTempFilename('.geojson')
        with open(json_file, 'w') as f:
            json.dump(self.geojson, f, separators=(',', ': '))

        gdal.UseExceptions()

        if self.model_file:
            # Assumes an image with no rotation
            gt=gdal.Info(self.model_file, format='json')['geoTransform']
            x_size, y_size=gdal.Info(self.model_file, format='json')['size']
            x_min=min(gt[0], gt[0] + x_size * gt[1])
            x_max=max(gt[0], gt[0] + x_size * gt[1])
            y_min=min(gt[3], gt[3] + y_size * gt[5])
            y_max=max(gt[3], gt[3] + y_size * gt[5])
            output_bounds=[x_min, y_min, x_max, y_max]
            x_res=gt[1]
            y_res=gt[5]
        else:
            output_bounds=None
            x_res=None
            y_res=None

        res = gdal.Rasterize(
            self.out_file,
            json_file,
            format='GTiff',
            outputBounds=output_bounds,
            initValues=-32767,  # Areas that are masked out
            burnValues=1,  # Areas that are NOT masked out
            xRes=x_res,
            yRes=y_res,
            outputSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            creationOptions=['COMPRESS=LZW'],
            callback=self.progress_callback
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None

    def progress_callback(self, fraction, message, data):
        if self.killed:
            return False
        else:
            self.progress.emit(100 * fraction)

            return True


class TranslateWorker(worker.AbstractWorker):
    def __init__(self, out_file, in_file):
        worker.AbstractWorker.__init__(self)

        self.out_file = out_file
        self.in_file = in_file

    def work(self):
        self.toggle_show_progress.emit(True)
        self.toggle_show_cancel.emit(True)

        gdal.UseExceptions()

        res = gdal.Translate(
            self.out_file,
            self.in_file,
            creationOptions=['COMPRESS=LZW'],
            callback=self.progress_callback
        )

        if res:
            return True
        else:
            return None

    def progress_callback(self, fraction, message, data):
        if self.killed:
            return False
        else:
            self.progress.emit(100 * fraction)

            return True
