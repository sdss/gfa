#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-02-27
# @Filename: test_post_process.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy
import pytest

from flicamera.camera import FLICamera


pytestmark = [pytest.mark.asyncio]


async def test_bias(camera: FLICamera):
    assert camera.exposure_meta.bias is None

    exposure = await camera.expose(0.0, image_type="bias")

    assert camera.exposure_meta.bias_file == str(exposure.filename)
    assert camera.exposure_meta.bias and camera.exposure_meta.bias > 0


async def test_dark(camera: FLICamera):

    await camera.expose(0.0, image_type="bias")
    dark = await camera.expose(2, image_type="dark")

    assert camera.exposure_meta.dark_file == str(dark.filename)
    assert isinstance(camera.exposure_meta.dark, numpy.ndarray)
