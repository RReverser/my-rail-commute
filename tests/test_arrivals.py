"""Tests for arrival-board support (mirrors the departures suite)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

from aioresponses import aioresponses
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import pytest

from custom_components.my_rail_commute.api import (
    InvalidStationError,
    NationalRailAPI,
)
from custom_components.my_rail_commute.const import (
    API_BASE_URL,
    CONF_DESTINATION,
    CONF_MAJOR_DELAY_THRESHOLD,
    CONF_MINOR_DELAY_THRESHOLD,
    CONF_NIGHT_UPDATES,
    CONF_NUM_SERVICES,
    CONF_ORIGIN,
    CONF_SEVERE_DELAY_THRESHOLD,
    CONF_TIME_WINDOW,
    CONF_TRACK_ARRIVALS,
    DEFAULT_MAJOR_DELAY_THRESHOLD,
    DEFAULT_MINOR_DELAY_THRESHOLD,
    DEFAULT_SEVERE_DELAY_THRESHOLD,
    STATUS_CANCELLED,
    STATUS_DELAYED,
    STATUS_ON_TIME,
)
from custom_components.my_rail_commute.coordinator import (
    NationalRailDataUpdateCoordinator,
)


@pytest.fixture(name="api_client")
async def api_client_fixture(aiohttp_session):
    """Create an API client with a real session."""
    return NationalRailAPI("test_api_key", aiohttp_session)


class TestGetArrivalBoard:
    """Tests for getting the arrival board."""

    async def test_get_arrival_board_success(
        self, api_client, arrival_board_response
    ):
        """Arrival board retrieval populates location/origin names and services."""
        with aioresponses() as mock:
            mock.get(
                f"{API_BASE_URL}/GetArrBoardWithDetails/HSL"
                f"?filterCrs=WAT&filterType=from&timeWindow=60&numRows=10",
                payload=arrival_board_response,
                status=200,
            )

            result = await api_client.get_arrival_board("HSL", "WAT")

            assert result["location_name"] == "Haslemere"
            assert result["origin_name"] == "London Waterloo"
            assert len(result["services"]) == 3
            assert result["generated_at"] == "2024-01-15T08:30:00"

    async def test_get_arrival_board_uppercase(self, api_client):
        """Lowercase CRS codes are normalised to uppercase."""
        with aioresponses() as mock:
            mock.get(
                f"{API_BASE_URL}/GetArrBoardWithDetails/HSL"
                f"?filterCrs=WAT&filterType=from&timeWindow=60&numRows=10",
                payload={"GetStationBoardResult": {"locationName": "Haslemere", "trainServices": []}},
                status=200,
            )

            await api_client.get_arrival_board("hsl", "wat")

    async def test_get_arrival_board_no_origin_filter(self, api_client):
        """Calling without an origin omits filterCrs/filterType."""
        with aioresponses() as mock:
            mock.get(
                f"{API_BASE_URL}/GetArrBoardWithDetails/HSL?timeWindow=60&numRows=10",
                payload={"GetStationBoardResult": {"locationName": "Haslemere", "trainServices": []}},
                status=200,
            )

            result = await api_client.get_arrival_board("HSL")
            assert result["origin_name"] is None

    async def test_get_arrival_board_invalid_station(self, api_client):
        """A 404 from the API surfaces as InvalidStationError."""
        with aioresponses() as mock:
            mock.get(
                f"{API_BASE_URL}/GetArrBoardWithDetails/XYZ"
                f"?filterCrs=WAT&filterType=from&timeWindow=60&numRows=10",
                status=404,
            )

            with pytest.raises(InvalidStationError):
                await api_client.get_arrival_board("XYZ", "WAT")


class TestParseArrivalService:
    """Tests for parsing a single arriving service."""

    async def test_parse_arrival_on_time(self, api_client):
        """An on-time arrival exposes sta as both scheduled and estimated arrival."""
        service = {
            "sta": "09:10",
            "eta": "On time",
            "platform": "2",
            "operator": "SWR",
            "serviceID": "svc1",
            "origin": [{"locationName": "London Waterloo", "crs": "WAT"}],
            "destination": [{"locationName": "Portsmouth Harbour", "crs": "PMH"}],
            "previousCallingPoints": [
                {
                    "callingPoint": [
                        {"locationName": "London Waterloo", "crs": "WAT", "st": "08:21", "et": "On time"},
                        {"locationName": "Guildford", "crs": "GLD", "st": "08:55", "et": "On time"},
                    ]
                }
            ],
        }

        result = api_client._parse_arrival_service(service, origin_crs="WAT")

        assert result["scheduled_arrival"] == "09:10"
        assert result["estimated_arrival"] == "09:10"
        assert result["status"] == STATUS_ON_TIME
        assert result["delay_minutes"] == 0
        assert result["is_cancelled"] is False
        assert result["origin"] == "London Waterloo"
        assert result["destination"] == "Portsmouth Harbour"
        # WAT is the requested origin, so the calling-points list starts there
        # and contains everything from origin onwards (no upstream cruft).
        assert result["calling_points"][0] == "London Waterloo"
        assert "Guildford" in result["calling_points"]
        # ``scheduled_departure`` is the departure time from the requested
        # origin, NOT the journey terminus.
        assert result["scheduled_departure"] == "08:21"
        assert result["expected_departure"] == "On time"

    async def test_parse_arrival_delayed(self, api_client):
        """A delayed arrival computes delay_minutes from sta vs eta."""
        service = {
            "sta": "09:40",
            "eta": "09:55",
            "platform": "1",
            "operator": "SWR",
            "serviceID": "svc2",
            "delayReason": "Signalling problems",
            "origin": [{"locationName": "London Waterloo", "crs": "WAT"}],
            "destination": [{"locationName": "Portsmouth Harbour", "crs": "PMH"}],
            "previousCallingPoints": [
                {
                    "callingPoint": [
                        {"locationName": "London Waterloo", "crs": "WAT", "st": "08:51", "et": "09:06"},
                    ]
                }
            ],
        }

        result = api_client._parse_arrival_service(service, origin_crs="WAT")

        assert result["status"] == STATUS_DELAYED
        assert result["delay_minutes"] == 15
        assert result["delay_reason"] == "Signalling problems"
        assert result["scheduled_arrival"] == "09:40"
        assert result["estimated_arrival"] == "09:55"
        assert result["scheduled_departure"] == "08:51"
        assert result["expected_departure"] == "09:06"

    async def test_parse_arrival_cancelled(self, api_client):
        """A cancelled arrival flips is_cancelled and surfaces the reason."""
        service = {
            "sta": "10:10",
            "eta": "Cancelled",
            "platform": "2",
            "operator": "SWR",
            "serviceID": "svc3",
            "cancelReason": "A fault on this train",
            "origin": [{"locationName": "London Waterloo", "crs": "WAT"}],
            "destination": [{"locationName": "Portsmouth Harbour", "crs": "PMH"}],
        }

        result = api_client._parse_arrival_service(service)

        assert result["status"] == STATUS_CANCELLED
        assert result["is_cancelled"] is True
        assert result["cancellation_reason"] == "A fault on this train"

    async def test_parse_arrival_falls_back_to_first_previous_stop(self, api_client):
        """Without origin_crs, scheduled_departure uses the earliest previous stop."""
        service = {
            "sta": "09:10",
            "eta": "On time",
            "platform": "2",
            "operator": "SWR",
            "serviceID": "svc4",
            "origin": [{"locationName": "London Waterloo", "crs": "WAT"}],
            "destination": [{"locationName": "Portsmouth Harbour", "crs": "PMH"}],
            "previousCallingPoints": [
                {
                    "callingPoint": [
                        {"locationName": "London Waterloo", "crs": "WAT", "st": "08:21", "et": "On time"},
                        {"locationName": "Guildford", "crs": "GLD", "st": "08:55", "et": "On time"},
                    ]
                }
            ],
        }

        result = api_client._parse_arrival_service(service)

        assert result["scheduled_departure"] == "08:21"
        assert result["calling_points"] == ["London Waterloo", "Guildford"]

    async def test_parse_arrival_calling_points_trim_upstream_of_origin(self, api_client):
        """Stops before the requested origin are dropped from calling_points."""
        service = {
            "sta": "09:10",
            "eta": "On time",
            "platform": "2",
            "operator": "SWR",
            "serviceID": "svc5",
            "origin": [{"locationName": "Far Origin", "crs": "FAR"}],
            "destination": [{"locationName": "Portsmouth Harbour", "crs": "PMH"}],
            "previousCallingPoints": [
                {
                    "callingPoint": [
                        {"locationName": "Far Origin", "crs": "FAR", "st": "07:00", "et": "On time"},
                        {"locationName": "London Waterloo", "crs": "WAT", "st": "08:21", "et": "On time"},
                        {"locationName": "Guildford", "crs": "GLD", "st": "08:55", "et": "On time"},
                    ]
                }
            ],
        }

        result = api_client._parse_arrival_service(service, origin_crs="WAT")

        assert "Far Origin" not in result["calling_points"]
        assert result["calling_points"][0] == "London Waterloo"
        assert result["scheduled_departure"] == "08:21"


class TestParseArrivalBoard:
    """Tests for parsing the arrival board envelope."""

    async def test_parse_arrival_board_with_services(self, api_client, arrival_board_response):
        result = api_client._parse_arrival_board(arrival_board_response, origin_crs="WAT")

        assert result["location_name"] == "Haslemere"
        assert result["origin_name"] == "London Waterloo"
        assert len(result["services"]) == 3
        assert result["services"][0]["status"] == STATUS_ON_TIME
        assert result["services"][1]["status"] == STATUS_DELAYED
        assert result["services"][1]["delay_minutes"] == 15
        assert result["services"][2]["status"] == STATUS_CANCELLED


def _make_arrivals_config():
    """Build a coordinator config tracking arrivals at HSL from WAT."""
    return {
        CONF_API_KEY: "test_key",
        CONF_ORIGIN: "WAT",
        CONF_DESTINATION: "HSL",
        CONF_TIME_WINDOW: 60,
        CONF_NUM_SERVICES: 3,
        CONF_NIGHT_UPDATES: True,
        CONF_SEVERE_DELAY_THRESHOLD: DEFAULT_SEVERE_DELAY_THRESHOLD,
        CONF_MAJOR_DELAY_THRESHOLD: DEFAULT_MAJOR_DELAY_THRESHOLD,
        CONF_MINOR_DELAY_THRESHOLD: DEFAULT_MINOR_DELAY_THRESHOLD,
        CONF_TRACK_ARRIVALS: True,
    }


class TestCoordinatorArrivalsMode:
    """Tests for the coordinator in arrivals mode."""

    async def test_track_arrivals_flag_set(self, hass: HomeAssistant):
        """The flag is read from config and exposed on the coordinator."""
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=dt_util.UTC)
        with patch(
            "custom_components.my_rail_commute.coordinator.dt_util.now",
            return_value=test_time,
        ):
            coordinator = NationalRailDataUpdateCoordinator(
                hass, AsyncMock(), _make_arrivals_config()
            )

        assert coordinator.track_arrivals is True

    async def test_arrivals_mode_calls_arrival_api(self, hass: HomeAssistant):
        """In arrivals mode the coordinator hits get_arrival_board, not get_departure_board."""
        test_time = datetime(2024, 1, 15, 7, 0, 0, tzinfo=dt_util.UTC)
        with patch(
            "custom_components.my_rail_commute.coordinator.dt_util.now",
            return_value=test_time,
        ):
            api = AsyncMock()
            api.get_arrival_board = AsyncMock(
                return_value={
                    "location_name": "Haslemere",
                    "origin_name": "London Waterloo",
                    "services": [],
                    "generated_at": "2024-01-15T07:00:00",
                    "nrcc_messages": [],
                }
            )
            api.get_departure_board = AsyncMock(
                side_effect=AssertionError("departures path must not be hit in arrivals mode")
            )

            coordinator = NationalRailDataUpdateCoordinator(
                hass, api, _make_arrivals_config()
            )
            data = await coordinator._async_update_data()

        api.get_arrival_board.assert_awaited_once()
        # In arrivals mode the watched station is the destination — its
        # name comes back from ``location_name`` on the response.
        assert coordinator.destination_name == "Haslemere"
        assert coordinator.origin_name == "London Waterloo"
        assert data["origin"] == "WAT"
        assert data["destination"] == "HSL"

    async def test_arrivals_mode_filters_already_arrived_trains(
        self, hass: HomeAssistant
    ):
        """Past-arrival trains are dropped, future-arrival trains are kept."""
        # Set the clock well past the first service so it's outside the
        # coordinator's grace period (default 5 minutes).
        test_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=dt_util.UTC)
        with patch(
            "custom_components.my_rail_commute.coordinator.dt_util.now",
            return_value=test_time,
        ):
            coordinator = NationalRailDataUpdateCoordinator(
                hass, AsyncMock(), _make_arrivals_config()
            )
            services = [
                # Arrived at 09:55 — 35 min in the past, well past grace
                {
                    "scheduled_arrival": "09:55",
                    "estimated_arrival": "09:55",
                    "scheduled_departure": "08:50",
                    "expected_departure": "08:50",
                    "is_cancelled": False,
                },
                # Arrives at 11:30 — still an hour in the future
                {
                    "scheduled_arrival": "11:30",
                    "estimated_arrival": "11:30",
                    "scheduled_departure": "10:30",
                    "expected_departure": "10:30",
                    "is_cancelled": False,
                },
            ]
            filtered = coordinator._filter_departed_trains(services)

        assert len(filtered) == 1
        assert filtered[0]["scheduled_arrival"] == "11:30"

    async def test_arrivals_mode_disabled_when_all_departures(self, hass: HomeAssistant):
        """The track_arrivals flag is forced off when all_departures is on."""
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=dt_util.UTC)
        config = _make_arrivals_config()
        config["all_departures"] = True
        config.pop(CONF_DESTINATION)
        with patch(
            "custom_components.my_rail_commute.coordinator.dt_util.now",
            return_value=test_time,
        ):
            coordinator = NationalRailDataUpdateCoordinator(hass, AsyncMock(), config)

        assert coordinator.track_arrivals is False
