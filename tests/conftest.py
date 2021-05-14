#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-12-18
# @Filename: conftest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib

import pytest

from sdsstools import read_yaml_file

from flicamera import FLICameraSystem
from flicamera.lib import LibFLI, LibFLIDevice
from flicamera.mock import MockFLIDevice, MockLibFLI, get_mock_camera_system


TEST_DATA = pathlib.Path(__file__).parent / "data/test_data.yaml"


@pytest.fixture(scope="session")
def config():
    """Gets the test configuration."""

    yield read_yaml_file(TEST_DATA)


@pytest.fixture
def mock_libfli(mocker):
    """Mocks the FLI library."""

    mocker.patch("ctypes.cdll.LoadLibrary", MockLibFLI)


@pytest.fixture
def libfli(mock_libfli, config):
    """Yields a LibFLI object with a mocked C libfli library."""

    libfli = LibFLI(simulation_mode=True)

    for camera in config["cameras"]:
        libfli.libc.devices.append(  # type:ignore
            MockFLIDevice(camera, status_params=config["cameras"][camera])
        )

    yield libfli

    LibFLIDevice._instances = {}


@pytest.fixture
def cameras(libfli):
    """Returns the connected cameras."""

    cameras = []

    for device in libfli.libc.devices:
        serial = device.state["serial"]
        cameras.append(libfli.get_camera(serial))

    yield cameras


@pytest.fixture
async def camera_system(mock_libfli, config):

    # devices = {}
    # for camera in config["cameras"]:
    #     devices[camera] = MockFLIDevice(camera, status_params=config["cameras"][camera])

    camera_system = await get_mock_camera_system(config, {})

    # camera_system = FLICameraSystem(camera_config=TEST_DATA, simulation_mode=True)
    # camera_system.lib.libc.devices = []  # type: ignore

    # for camera in config["cameras"]:
    #     device = MockFLIDevice(camera, status_params=config["cameras"][camera])
    #     camera_system.lib.libc.devices.append(device)

    camera_system.setup()
    for camera in camera_system.cameras:
        print(camera._device)

    yield camera_system

    LibFLIDevice._instances = {}

    for camera in camera_system.cameras:
        await camera.disconnect()

    await camera_system.disconnect()


@pytest.fixture
async def camera(camera_system):

    yield camera_system.cameras[0]
