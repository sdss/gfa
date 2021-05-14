#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-01-07
# @Filename: camera.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import time
import warnings
from glob import glob

from typing import Any, Dict, List, Optional, Tuple, Type

import astropy.io.fits
import astropy.time
import numpy
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder

from basecam import BaseCamera, CameraEvent, CameraSystem, Exposure
from basecam.exceptions import CameraConnectionError, CameraWarning, ExposureError
from basecam.mixins import CoolerMixIn, ExposureTypeMixIn, ImageAreaMixIn
from basecam.models import Card, Extension, HeaderModel

import flicamera
from flicamera.lib import LibFLI, LibFLIDevice
from flicamera.model import flicamera_model


__all__ = ["FLICameraSystem", "FLICamera"]


class ExposureMeta:
    """Bookkeeping for exposure data."""

    def __init__(self, camera: FLICamera):

        self.camera = camera

        self.mjd: int = 0
        self.dirname: Optional[str] = None

        self.bias_file: Optional[str] = None
        self.flat_file: Optional[str] = None
        self.dark_file: Optional[str] = None

        self.bias: Optional[float] = None
        self.flat: Optional[numpy.ndarray] = None
        self.dark: Optional[numpy.ndarray] = None

    def reset(self):
        """Reset values."""

        self.mjd: int = 0
        self.dirname = None

        self.bias_file = None
        self.flat_file = None
        self.dark_file = None

        self.bias = None
        self.flat = None
        self.dark = None

    async def update_images(self):
        """Loops over existing images and determines bias, flat, and dark values."""

        self.reset()

        image_namer = self.camera.image_namer

        self.mjd = int(astropy.time.Time.now().mjd)
        self.dirname = str(image_namer.get_dirname())

        basename: str = image_namer.basename.format(camera=self.camera, num=0)
        basename = basename.replace("0000", "*")

        images = glob(os.path.join(self.dirname, basename))
        images = list(sorted(images, reverse=True))

        for image in images:
            hdul: Any = astropy.io.fits.open(image)
            if hdul["RAW"].header["MJD"] != self.mjd:
                warnings.warn(
                    "Inconsistent MJD. Cannot determine bias, flat, or dark.",
                    CameraWarning,
                )
                return

            header = hdul["RAW"].header
            image_type: str = header["IMAGETYP"]

            if image_type == "bias":
                self.bias_file = image
                self.bias = numpy.median(hdul["RAW"].data)
            elif image_type == "flat":
                if "PROC" in hdul:
                    self.flat_file = image
                    self.flat = hdul["PROC"].data
            elif image_type == "dark":
                if "PROC" in hdul:
                    self.dark_file = image
                    self.dark = hdul["PROC"].data
            else:
                if "BIASFILE" in header and header["BIASFILE"] != "":
                    self.bias_file = os.path.join(self.dirname, header["BIASFILE"])
                    self.bias = numpy.median(astropy.io.fits.getdata(self.bias_file))
                if "FLATFILE" in header and header["FLATFILE"] != "":
                    flat_file = os.path.join(self.dirname, header["FLATFILE"])
                    flat: Any = astropy.io.fits.open(self.flat_file)
                    if "PROC" in flat:
                        self.flat_file = flat_file
                        self.flat = flat["PROC"].data
                if "DARKFILE" in header and header["DARKFILE"] != "":
                    dark_file = os.path.join(self.dirname, header["DARKFILE"])
                    dark: Any = astropy.io.fits.open(self.dark_file)
                    if "PROC" in dark:
                        self.dark_file = dark_file
                        self.dark = dark["PROC"].data / dark["PROC"].header["EXPTIME"]

            if self.bias and self.flat and self.dark:
                break


class FLICamera(BaseCamera, ExposureTypeMixIn, CoolerMixIn, ImageAreaMixIn):
    """A FLI camera."""

    camera_system: FLICameraSystem

    _device: Optional[LibFLIDevice] = None
    fits_model = flicamera_model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gain: float = self.camera_params.get("gain", -999)
        self.read_noise: float = self.camera_params.get("read_noise", -999)

        self.observatory: str = flicamera.OBSERVATORY
        if self.observatory in flicamera.config["pixel_scale"]:
            self.pixel_scale: float = flicamera.config["pixel_scale"][self.observatory]
        else:
            self.pixel_scale: float = -999.0

        self.exposure_meta = ExposureMeta(self)

    async def _connect_internal(self, **conn_params):
        """Internal method to connect the camera."""

        serial = conn_params.get("serial", conn_params.get("uid", self.uid))

        if serial is None:
            raise CameraConnectionError("unknown serial number.")

        self._device = self.camera_system.lib.get_camera(serial)

        if self._device is None:
            raise CameraConnectionError(f"cannot find camera with serial {serial}.")

        # Update image information. Schedule as a task because this can be time
        # consuming as it needs to loop over potentially many images.
        asyncio.create_task(self.exposure_meta.update_images())

    def _status_internal(self) -> Dict[str, Any]:
        """Gets a dictionary with the status of the camera.

        Returns
        -------
        status
            A dictionary with status values from the camera (e.g.,
            temperature, cooling status, firmware information, etc.)
        """

        device = self._device
        device._update_temperature()

        return dict(
            model=device.model,
            serial=device.serial,
            fwrev=device.fwrev,
            hwrev=device.hwrev,
            hbin=device.hbin,
            vbin=device.vbin,
            visible_area=device.get_visible_area(),
            image_area=device.area,
            temperature_ccd=device._temperature["CCD"],
            temperature_base=device._temperature["base"],
            exposure_time_left=device.get_exposure_time_left(),
            cooler_power=device.get_cooler_power(),
        )

    async def _expose_internal(
        self,
        exposure: Exposure,
        **kwargs,
    ) -> Exposure:
        """Internal method to handle camera exposures."""

        if exposure.exptime is None:
            raise ExposureError("Exposure time not set.")

        TIMEOUT = 5

        device = self._device

        device.cancel_exposure()

        device.set_exposure_time(exposure.exptime)

        image_type = exposure.image_type
        frametype = "dark" if image_type in ["dark", "bias"] else "normal"

        device.start_exposure(frametype)

        exposure.obstime = astropy.time.Time.now()
        self.notify(CameraEvent.EXPOSURE_INTEGRATING)

        start_time = time.time()

        time_left = exposure.exptime

        while True:

            await asyncio.sleep(time_left)

            time_left = device.get_exposure_time_left() / 1000.0

            if time_left == 0:
                self.notify(CameraEvent.EXPOSURE_READING)
                array = await self.loop.run_in_executor(None, device.read_frame)
                exposure.data = array
                break

            if time.time() - start_time > exposure.exptime + TIMEOUT:
                raise ExposureError("timeout while waiting for exposure to finish.")

        return exposure

    async def _post_process_internal(self, exposure: Exposure, **kwargs) -> Exposure:
        """Post-processes the exposure."""

        # TODO: this is an initial, quite simple version. Needs to be improved.

        self.notify(CameraEvent.EXPOSURE_POST_PROCESSING)

        mjd = astropy.time.Time(exposure.obstime).mjd
        if mjd != self.exposure_meta.mjd:
            await self.exposure_meta.update_images()

        meta = self.exposure_meta

        assert exposure.exptime is not None, "Exptime cannot be null."
        assert exposure.filename is not None, "filename has not been set."

        image_type = exposure.image_type
        filename = exposure.filename

        fits_model = exposure.fits_model
        raw_extension: Extension = fits_model[0]
        proc_header: HeaderModel = HeaderModel(raw_extension.header_model[:])

        extensions = []

        failed = False
        error = None

        if image_type == "bias":
            meta.bias_file = filename
            meta.bias = numpy.median(exposure.data)
            proc_header.append(Card("COMMENT", f"BIAS = {meta.bias:.2f}"))
            extensions.append(
                Extension(
                    data=False,
                    header_model=proc_header,
                    name="PROC",
                )
            )

        elif image_type == "dark":
            if not meta.bias:
                failed = True
                error = "Cannot find bias image"
            else:
                print("hi")
                meta.dark_file = filename
                ddata = exposure.data.copy().astype("float32")
                ddata -= meta.bias
                meta.dark = ddata / exposure.exptime

                proc_header.append(Card("COMMENT", f"BIAS = {meta.bias:.2f}"))
                extensions.append(
                    Extension(
                        data=ddata,
                        header_model=proc_header,
                        name="PROC",
                        compressed="RICE_1",
                    )
                )

        elif image_type == "flat":
            if not meta.bias or not meta.dark:
                failed = True
                error = "Cannot find bias or dark images"
            else:
                meta.bias_file = filename
                fdata = exposure.data.copy().astype("float32")
                fdata -= meta.bias
                fdata -= meta.dark * exposure.exptime
                fdata /= numpy.median(fdata)
                meta.flat = fdata

                proc_header.append(Card("COMMENT", f"BIAS = {meta.bias:.2f}"))
                extensions.append(
                    Extension(
                        data=fdata,
                        header_model=proc_header,
                        name="PROC",
                        compressed="RICE_1",
                    )
                )

        elif image_type == "object":
            if not meta.bias or not meta.dark or not meta.flat:
                failed = True
                error = "Cannot find bias, dark, or flat images"
            else:
                meta.bias_file = filename
                pdata = exposure.data.copy().astype("float32")
                pdata -= meta.bias
                pdata -= meta.dark * exposure.exptime
                pdata /= meta.flat

                proc_header.append(Card("COMMENT", f"BIAS = {meta.bias:.2f}"))
                extensions.append(
                    Extension(
                        data=pdata,
                        header_model=proc_header,
                        name="PROC",
                        compressed="RICE_1",
                    )
                )

                __, __, std = sigma_clipped_stats(pdata, sigma=3.0)
                daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
                sources = daofind(pdata)

                fits_model.append(astropy.io.fits.BinTableHDU(sources))

        if failed:
            self.notify(CameraEvent.EXPOSURE_POST_PROCESS_FAILED, dict(error=error))
            return exposure

        exposure.fits_model += extensions

        self.notify(CameraEvent.EXPOSURE_POST_PROCESS_DONE)

        return exposure

    async def _get_temperature_internal(self) -> float:
        """Internal method to get the camera temperature."""

        self._device._update_temperature()
        return self._device._temperature["CCD"]

    async def _set_temperature_internal(self, temperature: float):
        """Internal method to set the camera temperature."""

        self._device.set_temperature(temperature)

    async def _get_image_area_internal(self) -> Tuple[int, int, int, int]:
        """Internal method to return the image area."""

        area = self._device.area

        # Convert from (ul_x, ul_y, lr_x, lr_y) to (x0, x1, y0, y1)
        area = (area[0], area[2], area[1], area[3])

        return area

    async def _set_image_area_internal(
        self,
        area: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Internal method to set the image area."""

        if area:
            # Convert from (x0, x1, y0, y1) to (ul_x, ul_y, lr_x, lr_y)
            area = (area[0], area[2], area[1], area[3])

        self._device.set_image_area(area)

    async def _get_binning_internal(self) -> Tuple[int, int]:
        """Internal method to return the binning."""

        return (self._device.hbin, self._device.vbin)

    async def _set_binning_internal(self, hbin, vbin):
        """Internal method to set the binning."""

        self._device.set_binning(hbin, vbin)


class FLICameraSystem(CameraSystem[FLICamera]):
    """FLI camera system."""

    __version__ = flicamera.__version__

    camera_class = FLICamera

    def __init__(self, *args, simulation_mode=False, **kwargs):

        self.camera_class: Type[FLICamera] = kwargs.pop("camera_system", FLICamera)
        self.lib = LibFLI(simulation_mode=simulation_mode)

        super().__init__(*args, **kwargs)

    def list_available_cameras(self) -> List[str]:

        # These are camera devices, not UIDs. They can change as cameras
        # are replugged or moved to a different computer.
        devices_id = self.lib.list_cameras()

        # Get the serial number as UID.
        serial_numbers = []
        for device_id in devices_id:
            device = LibFLIDevice(device_id, self.lib.libc)
            serial_numbers.append(device.serial)

        return serial_numbers
