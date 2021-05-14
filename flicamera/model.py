#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-02-12
# @Filename: model.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import abc

from typing import Any, Dict, List, Optional, Tuple, Union

from basecam.exposure import Exposure
from basecam.models import (
    Card,
    CardGroup,
    Extension,
    FITSModel,
    HeaderModel,
    MacroCard,
    WCSCards,
)
from clu.legacy.types.pvt import PVT

import flicamera


__all__ = ["flicamera_model"]


MacroCardReturnType = List[Union[Tuple[str, Any], Tuple[str, Any, str], Card]]


def pvt2pos(tup):
    pvt = PVT(*tup)
    return pvt.getPos()


# TODO: this should be moved to CLU.


class TronModelCards(MacroCard, metaclass=abc.ABCMeta):
    model = None
    model_name: Optional[str] = None

    def macro(self, exposure: Exposure, context: Dict[str, Any] = {}):

        try:
            self.model = context["__actor__"].tron.models[self.model_name]
        except (KeyError, AttributeError):
            self.model = None

        return self._cards(exposure, context=context)

    @abc.abstractmethod
    def _cards(
        self,
        exposure: Exposure,
        context: Dict[str, Any],
    ) -> MacroCardReturnType:
        raise NotImplementedError

    def get(self, key, idx=0, default="NaN", cnv=None):
        if not self.model:
            return default

        try:
            value = self.model[key].value
            if idx is not None:
                value = value[idx]
            if cnv:
                value = cnv(value)
            return value
        except BaseException:
            return default


class APOTCCCards(TronModelCards):
    """Return a list of Cards describing the APO TCC state."""

    name = "APO TCC Cards"
    model_name = "tcc"

    def _cards(
        self,
        exposure: Exposure,
        context: Dict[str, Any] = {},
    ) -> MacroCardReturnType:

        cards: MacroCardReturnType = []

        objSysName = self.get("objSys", 0, default="UNKNOWN")
        cards.append(("OBJSYS", objSysName, "The TCC objSys"))

        # ObjSys
        if objSysName in ("None", "Mount", "Obs", "Phys", "Inst"):
            cards += [
                ("RA", "NaN", "Telescope is not tracking the sky"),
                ("DEC", "NaN", "Telescope is not tracking the sky"),
                ("RADEG", "NaN", "Telescope is not tracking the sky"),
                ("DECDEG", "NaN", "Telescope is not tracking the sky"),
                ("SPA", "NaN", "Telescope is not tracking the sky"),
            ]
        else:
            cards += [
                (
                    "RA",
                    self.get("objNetPos", 0, cnv=pvt2pos),
                    "RA of telescope boresight (deg)",
                ),
                (
                    "DEC",
                    self.get("objNetPos", 1, cnv=pvt2pos),
                    "Dec of telescope boresight (deg)",
                ),
                (
                    "RADEG",
                    self.get("objPos", 0, cnv=pvt2pos),
                    "RA of telescope pointing (deg)",
                ),
                (
                    "DECDEG",
                    self.get("objPos", 1, cnv=pvt2pos),
                    "Dec of telescope pointing (deg)",
                ),
                (
                    "SPA",
                    self.get("spiderInstAng", 0, cnv=pvt2pos),
                    "TCC SpiderInstAng",
                ),
            ]

        # Rotator
        cards += [
            (
                "ROTTYPE",
                self.get("rotType", 0, cnv=str, default="UNKNOWN"),
                "Rotator request type",
            ),
            (
                "ROTPOS",
                self.get("rotPos", 0, cnv=pvt2pos),
                "Rotator request type",
            ),
        ]

        # Offsets
        offsets = (
            ("boresight", "BOREOFF", "TCC Boresight offset, deg", False),
            ("objArcOff", "ARCOFF", "TCC ObjArcOff, deg", False),
            ("calibOff", "CALOFF", "TCC CalibOff, deg", True),
            ("guideOff", "GUIDOFF", "TCC GuideOff, deg", True),
        )
        for tcc_key, key, comment, do_rot in offsets:
            cards.append(
                (
                    key + "X",
                    self.get(tcc_key, 0, cnv=pvt2pos),
                    comment,
                )
            )
            cards.append(
                (
                    key + "Y",
                    self.get(tcc_key, 1, cnv=pvt2pos),
                    comment,
                )
            )
            if do_rot:
                cards.append(
                    (
                        key + "R",
                        self.get(tcc_key, 2, cnv=pvt2pos),
                        comment,
                    )
                )

        # Alt/Az
        cards += [
            (
                "AZ",
                self.get("axePos", 0, cnv=float),
                "Azimuth axis pos. (approx, deg)",
            ),
            (
                "ALT",
                self.get("axePos", 1, cnv=float),
                "Altitude axis pos. (approx, deg)",
            ),
            (
                "IPA",
                self.get("axePos", 2, cnv=float),
                "Rotator axis pos. (approx, deg)",
            ),
            (
                "FOCUS",
                self.get("secFocus", 0, cnv=float),
                "User-specified focus offset (um)",
            ),
        ]

        # M1/2 orientation
        orientNames = ("piston", "xtilt", "ytilt", "xtran", "ytran", "zrot")
        for mirror in (1, 2):
            for idx, name in enumerate(orientNames):
                cards.append(
                    (
                        ("M1" if mirror == 1 else "M2") + orientNames[idx],
                        self.get(
                            "primOrient" if mirror == 1 else "secOrient",
                            idx,
                            cnv=float,
                        ),
                        "TCC " + "PrimOrient" if mirror == 1 else "SecOrient",
                    )
                )

        cards.append(
            (
                "SCALE",
                self.get("scaleFac", 0, cnv=float),
                "User-specified scale factor",
            )
        )

        return cards


class LampCards(TronModelCards):
    """ Return a list of Cards describing the MCP state. """

    name = "Lamp Cards"
    model_name = "mcp"

    def _cnvLampCard(self, lamps):
        """Convert the MCP lamp keyword to what we want.

        ``lamps`` here are a list of True/False for each lamp.
        """
        conv = []
        for ll in lamps:
            try:
                conv.append(str(int(ll)))
            except Exception:
                conv.append("X")
        return " ".join(conv)

    def _cnvFFSCard(self, petals):
        """Convert the mcp.ffsStatus keyword to what we want.

        ``petals`` is a list of ``"01"`` or ``"10"`` for open and closed respectively.
        """
        ffDict = {"01": "1", "10": "0"}
        return " ".join([str(ffDict.get(p, "X")) for p in petals])

    def _cards(
        self,
        exposure: Exposure,
        context: Dict[str, Any] = {},
    ) -> MacroCardReturnType:

        cards: MacroCardReturnType = []

        for lamp_key in ("ffLamp", "neLamp", "hgCdLamp"):
            card_name = lamp_key[:-4].upper()
            card = (
                card_name,
                self.get(
                    lamp_key,
                    idx=None,
                    cnv=self._cnvLampCard,
                    default="X X X X",
                ),
                f"{card_name} lamps 1:on 0:0ff",
            )
            cards.append(card)

        cards.append(
            (
                "FFS",
                self.get(
                    "ffsStatus",
                    idx=None,
                    cnv=self._cnvFFSCard,
                    default="X X X X X X X X",
                ),
                "Flatfield Screen 1:closed 0:open",
            )
        )

        return cards


class APOCards(TronModelCards):
    """ Return a list of Cards describing the APO actor state. """

    name = "Lamp Cards"
    model_name = "apo"

    def _cards(
        self,
        exposure: Exposure,
        context: Dict[str, Any] = {},
    ) -> MacroCardReturnType:

        cards: MacroCardReturnType = []

        cards.append(
            (
                "V_APO",
                self.get("version", default="Unknown"),
                "Version of the current apoActor",
            )
        )

        keys = (
            ("pressure", None, float),
            ("windd", None, float),
            ("winds", None, float),
            ("gustd", None, float),
            ("gusts", None, float),
            ("airTempPT", "airtemp", float),
            ("dpTempPT", "dewpoint", float),
            ("truss25m", "trustemp", float),
            # ('dpErrPT', None, str),
            ("humidity", None, float),
            ("dusta", None, float),
            ("dustb", None, float),
            ("windd25m", None, float),
            ("winds25m", None, float),
        )

        for key_name, card_name, cnv in keys:
            if not card_name:
                card_name = key_name
            card_name = card_name.upper()
            card = (card_name, self.get(key_name, default="NaN"))
            cards.append(card)

        return cards


window_group = CardGroup(
    [
        Card(
            "BEGX",
            "{__camera__._device.area[0]}",
            "Window start pixel in X",
            default=-999,
        ),
        Card(
            "BEGY",
            "{__camera__._device.area[1]}",
            "Window start pixel in Y",
            default=-999,
        ),
        Card(
            "ENDX",
            "{__camera__._device.area[2]}",
            "Window end pixel in X",
            default=-999,
        ),
        Card(
            "EDNY",
            "{__camera__._device.area[3]}",
            "Window end pixel in Y",
            default=-999,
        ),
        Card(
            "BINX",
            "{__camera__._device.hbin}",
            "Binning in X",
            default=-999,
        ),
        Card(
            "BINY",
            "{__camera__._device.vbin}",
            "Binning in Y",
            default=-999,
        ),
    ],
    name="window",
    use_group_title=False,
)


raw_header_model = HeaderModel(
    [
        "CAMNAME",
        "VCAM",
        "IMAGETYP",
        "EXPTIME",
        "EXPTIMEN",
        "STACK",
        "STACKFUN",
        Card(
            "TIMESYS",
            "TAI",
            "Time reference system",
        ),
        Card(
            "DATE-OBS",
            "{__exposure__.obstime.tai}",
            "Time of the start of the exposure [TAI]",
        ),
        Card(
            "MJD",
            "int(__exposure__.obstime.mjd)",
            "Modified Julian Day of the exposure",
            evaluate=True,
        ),
        Card(
            "CCDTEMP",
            "{__camera__.status[temperature_ccd]}",
            "Degrees C",
            default=-999.0,
        ),
        window_group,
        Card(
            "GAIN",
            "{__camera__.gain}",
            "The CCD gain [e-/ADUs]",
            default=-999.0,
            type=float,
        ),
        Card(
            "READNOIS",
            "{__camera__.read_noise}",
            "The CCD read noise [ADUs]",
            default=-999.0,
            type=float,
        ),
        Card(
            "PIXELSC",
            "{__camera__.pixel_scale}",
            "Scale of unbinned pixel on sky [arcsec/pix]",
            default=-999.0,
            type=float,
        ),
        WCSCards(),
        Card(
            "OBSERVAT",
            "{__camera__.observatory}",
            "Observatory",
            default="",
        ),
        APOTCCCards() if flicamera.OBSERVATORY == "APO" else None,
        LampCards() if flicamera.OBSERVATORY == "APO" else None,
        APOCards() if flicamera.OBSERVATORY == "APO" else None,
    ]
)

flicamera_model = FITSModel(
    [
        Extension(
            data="raw",
            header_model=raw_header_model,
            name="RAW",
            compressed="GZIP_2",
        )
    ]
)
