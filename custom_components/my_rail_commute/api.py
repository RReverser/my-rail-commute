"""Rail API client for Live Departure Boards."""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta
import logging
import re
from typing import Any

import aiohttp
from aiohttp import ClientError, ClientResponseError

from .const import (
    API_BASE_URL,
    API_TIMEOUT,
    ERROR_API_UNAVAILABLE,
    ERROR_AUTH,
    ERROR_INVALID_STATION,
    ERROR_NETWORK,
    ERROR_RATE_LIMIT,
    STATUS_CANCELLED,
    STATUS_DELAYED,
    STATUS_ON_TIME,
    USER_AGENT,
)

_LOGGER = logging.getLogger(__name__)

_TIME_FORMAT_RE = re.compile(r"^\d{2}:\d{2}$")

# Rate limit configuration
# These defaults are conservative; adjust based on actual API limits
DEFAULT_RATE_LIMIT_PER_MINUTE = 10
DEFAULT_RATE_LIMIT_PER_HOUR = 100
RATE_LIMIT_THROTTLE_THRESHOLD = 0.8  # Throttle at 80% of limit


class NationalRailAPIError(Exception):
    """Base exception for Rail API errors."""


class AuthenticationError(NationalRailAPIError):
    """Authentication failed."""


class InvalidStationError(NationalRailAPIError):
    """Invalid station code."""


class RateLimitError(NationalRailAPIError):
    """API rate limit exceeded."""


class NationalRailAPI:
    """Rail API client for Live Departure Boards."""

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        rate_limit_per_hour: int = DEFAULT_RATE_LIMIT_PER_HOUR,
    ) -> None:
        """Initialize the API client.

        Args:
            api_key: Rail Data Marketplace API key
            session: aiohttp client session
            rate_limit_per_minute: Maximum requests per minute
            rate_limit_per_hour: Maximum requests per hour
        """
        self._api_key = api_key
        self._session = session
        self._base_url = API_BASE_URL
        self._headers = {
            "x-apikey": api_key,
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

        # Rate limit tracking
        self._rate_limit_per_minute = rate_limit_per_minute
        self._rate_limit_per_hour = rate_limit_per_hour
        self._call_timestamps: deque[datetime] = deque()  # Sliding window of API call times

    def _clean_old_calls(self, window_minutes: int) -> None:
        """Remove API call timestamps older than the specified window.

        Args:
            window_minutes: Time window in minutes to keep
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        while self._call_timestamps and self._call_timestamps[0] < cutoff_time:
            self._call_timestamps.popleft()

    def _get_calls_in_window(self, window_minutes: int) -> int:
        """Get the number of API calls within the specified time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Number of calls within the window
        """
        self._clean_old_calls(window_minutes)
        return len(self._call_timestamps)

    def _check_rate_limit_proximity(self) -> tuple[bool, float]:
        """Check if we're approaching rate limits.

        Returns:
            Tuple of (should_throttle, wait_seconds)
            - should_throttle: True if we should wait before making the next call
            - wait_seconds: How long to wait (0 if no throttling needed)
        """
        # Check per-minute limit
        calls_per_minute = self._get_calls_in_window(1)
        minute_threshold = int(self._rate_limit_per_minute * RATE_LIMIT_THROTTLE_THRESHOLD)

        if calls_per_minute >= self._rate_limit_per_minute:
            # At or over limit - calculate wait time until oldest call expires
            if self._call_timestamps:
                oldest_call = self._call_timestamps[0]
                wait_until = oldest_call + timedelta(minutes=1)
                wait_seconds = max(0, (wait_until - datetime.now()).total_seconds())
                _LOGGER.warning(
                    "Rate limit reached: %s/%s calls per minute. Waiting %.1f seconds.",
                    calls_per_minute,
                    self._rate_limit_per_minute,
                    wait_seconds,
                )
                return True, wait_seconds
        elif calls_per_minute >= minute_threshold:
            # Approaching limit - add small delay to spread out requests
            wait_seconds = 60.0 / self._rate_limit_per_minute
            _LOGGER.info(
                "Approaching rate limit: %s/%s calls per minute (threshold: %s). "
                "Adding %.1f second delay.",
                calls_per_minute,
                self._rate_limit_per_minute,
                minute_threshold,
                wait_seconds,
            )
            return True, wait_seconds

        # Check per-hour limit
        calls_per_hour = self._get_calls_in_window(60)
        hour_threshold = int(self._rate_limit_per_hour * RATE_LIMIT_THROTTLE_THRESHOLD)

        if calls_per_hour >= self._rate_limit_per_hour:
            # At or over limit
            if self._call_timestamps:
                # Find oldest call and wait until it expires from the hour window
                oldest_call = self._call_timestamps[0]
                wait_until = oldest_call + timedelta(hours=1)
                wait_seconds = max(0, (wait_until - datetime.now()).total_seconds())
                _LOGGER.warning(
                    "Hourly rate limit reached: %s/%s calls per hour. Waiting %.1f seconds.",
                    calls_per_hour,
                    self._rate_limit_per_hour,
                    wait_seconds,
                )
                return True, wait_seconds
        elif calls_per_hour >= hour_threshold:
            # Approaching hourly limit - add delay
            wait_seconds = 3600.0 / self._rate_limit_per_hour
            _LOGGER.info(
                "Approaching hourly rate limit: %s/%s calls per hour (threshold: %s). "
                "Adding %.1f second delay.",
                calls_per_hour,
                self._rate_limit_per_hour,
                hour_threshold,
                wait_seconds,
            )
            return True, wait_seconds

        return False, 0.0

    async def _throttle_if_needed(self) -> None:
        """Proactively throttle requests if approaching rate limits."""
        should_throttle, wait_seconds = self._check_rate_limit_proximity()
        if should_throttle and wait_seconds > 0:
            _LOGGER.debug("Throttling request for %.1f seconds", wait_seconds)
            await asyncio.sleep(wait_seconds)

    def _record_api_call(self) -> None:
        """Record a successful API call for rate limit tracking."""
        self._call_timestamps.append(datetime.now())
        # Keep only last hour of data to prevent unbounded growth
        self._clean_old_calls(60)

        # Log current usage periodically (every 10th call)
        if len(self._call_timestamps) % 10 == 0:
            calls_per_minute = self._get_calls_in_window(1)
            calls_per_hour = self._get_calls_in_window(60)
            _LOGGER.debug(
                "API usage: %s/%s per minute, %s/%s per hour",
                calls_per_minute,
                self._rate_limit_per_minute,
                calls_per_hour,
                self._rate_limit_per_hour,
            )

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Make an API request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            retry_count: Current retry attempt
            max_retries: Maximum number of retries

        Returns:
            Parsed JSON response

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            NationalRailAPIError: For other API errors
        """
        # Proactively check and throttle if approaching rate limits
        await self._throttle_if_needed()

        url = f"{self._base_url}/{endpoint}"

        try:
            async with asyncio.timeout(API_TIMEOUT):
                async with self._session.get(
                    url, headers=self._headers, params=params
                ) as response:
                    # Log the full request details for debugging
                    _LOGGER.debug(
                        "API request: %s %s (status: %s)",
                        "GET",
                        url,
                        response.status,
                    )

                    # Handle different status codes
                    if response.status == 401 or response.status == 403:
                        _LOGGER.error("Authentication failed with status %s", response.status)
                        raise AuthenticationError(ERROR_AUTH)

                    if response.status == 429:
                        # Rate limit exceeded
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff
                            _LOGGER.warning(
                                "Rate limit exceeded, retrying in %s seconds",
                                wait_time,
                            )
                            await asyncio.sleep(wait_time)
                            return await self._request(
                                endpoint, params, retry_count + 1, max_retries
                            )
                        raise RateLimitError(ERROR_RATE_LIMIT)

                    if response.status == 400:
                        _LOGGER.error(
                            "Invalid request (400) for endpoint: %s. "
                            "This typically indicates an invalid CRS station code",
                            endpoint,
                        )
                        raise InvalidStationError(ERROR_INVALID_STATION)

                    if response.status == 404:
                        _LOGGER.error("Station not found (404) for endpoint: %s", endpoint)
                        raise InvalidStationError(ERROR_INVALID_STATION)

                    # Handle server errors (500+) with retry
                    if response.status >= 500:
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count
                            _LOGGER.warning(
                                "Server error %s, retrying in %s seconds (attempt %s/%s)",
                                response.status,
                                wait_time,
                                retry_count + 1,
                                max_retries,
                            )
                            await asyncio.sleep(wait_time)
                            return await self._request(
                                endpoint, params, retry_count + 1, max_retries
                            )
                        _LOGGER.error("API server error %s after %s retries", response.status, max_retries)
                        raise NationalRailAPIError(f"API server error {response.status}: {ERROR_API_UNAVAILABLE}")

                    # Check for other non-success status codes
                    response.raise_for_status()

                    try:
                        data = await response.json(content_type=None)
                    except (ValueError, aiohttp.ContentTypeError) as err:
                        _LOGGER.error("Invalid JSON response from API: %s", err)
                        raise NationalRailAPIError(ERROR_API_UNAVAILABLE) from err

                    # Record successful API call for rate limit tracking
                    self._record_api_call()

                    return data

        except TimeoutError as err:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                _LOGGER.warning(
                    "Request timeout, retrying in %s seconds (attempt %s/%s)",
                    wait_time,
                    retry_count + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
                return await self._request(
                    endpoint, params, retry_count + 1, max_retries
                )
            _LOGGER.error("Request timeout after %s retries", max_retries)
            raise NationalRailAPIError(ERROR_NETWORK) from err

        except ClientResponseError as err:
            # This should rarely be hit now since we handle status codes explicitly
            _LOGGER.error("Unhandled HTTP error %s: %s", err.status, err.message)
            raise NationalRailAPIError(f"HTTP error {err.status}: {ERROR_API_UNAVAILABLE}") from err

        except ClientError as err:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                _LOGGER.warning("Network error, retrying in %s seconds", wait_time)
                await asyncio.sleep(wait_time)
                return await self._request(
                    endpoint, params, retry_count + 1, max_retries
                )
            _LOGGER.error("Network error: %s", err)
            raise NationalRailAPIError(ERROR_NETWORK) from err

    async def get_departure_board(
        self,
        origin_crs: str,
        destination_crs: str | None = None,
        time_window: int = 60,
        num_rows: int = 10,
    ) -> dict[str, Any]:
        """Get departure board for a route.

        Args:
            origin_crs: Origin station CRS code (3 letters)
            destination_crs: Destination station CRS code (3 letters), or None for all departures
            time_window: Time window in minutes
            num_rows: Number of services to retrieve

        Returns:
            Departure board data with services

        Raises:
            InvalidStationError: If station codes are invalid
            NationalRailAPIError: For other API errors
        """
        _LOGGER.debug(
            "Fetching departure board: %s -> %s (window: %s mins, rows: %s)",
            origin_crs,
            destination_crs or "ALL",
            time_window,
            num_rows,
        )

        # Use path parameters for the CRS code and query parameters for filters
        endpoint = f"GetDepBoardWithDetails/{origin_crs.upper()}"
        params: dict[str, Any] = {
            "timeWindow": time_window,
            "numRows": num_rows,
        }
        if destination_crs:
            params["filterCrs"] = destination_crs.upper()

        try:
            data = await self._request(endpoint, params)
            return self._parse_departure_board(data, destination_crs)
        except InvalidStationError:
            raise
        except NationalRailAPIError as err:
            _LOGGER.error("Failed to get departure board: %s", err)
            raise

    async def get_arrival_board(
        self,
        destination_crs: str,
        origin_crs: str | None = None,
        time_window: int = 60,
        num_rows: int = 10,
    ) -> dict[str, Any]:
        """Get arrival board for a route.

        Mirrors get_departure_board() but queries trains arriving at
        ``destination_crs``, optionally filtered by the train's origin
        ``origin_crs``. Output uses the same shape as get_departure_board so
        the coordinator and sensors stay direction-agnostic — ``scheduled_*``
        / ``expected_*`` fields refer to the watched (destination) station's
        arrival time.

        Args:
            destination_crs: Station to watch arrivals at (3 letters)
            origin_crs: Optional origin station CRS to filter by
            time_window: Time window in minutes
            num_rows: Number of services to retrieve

        Returns:
            Arrival board data with services

        Raises:
            InvalidStationError: If station codes are invalid
            NationalRailAPIError: For other API errors
        """
        _LOGGER.debug(
            "Fetching arrival board: %s <- %s (window: %s mins, rows: %s)",
            destination_crs,
            origin_crs or "ALL",
            time_window,
            num_rows,
        )

        endpoint = f"GetArrBoardWithDetails/{destination_crs.upper()}"
        params: dict[str, Any] = {
            "timeWindow": time_window,
            "numRows": num_rows,
        }
        if origin_crs:
            params["filterCrs"] = origin_crs.upper()
            # filterType=from restricts to services whose journey originates
            # (or has previously called) at the filter station.
            params["filterType"] = "from"

        try:
            data = await self._request(endpoint, params)
            return self._parse_arrival_board(data, origin_crs)
        except InvalidStationError:
            raise
        except NationalRailAPIError as err:
            _LOGGER.error("Failed to get arrival board: %s", err)
            raise

    def _parse_departure_board(self, data: dict[str, Any], destination_crs: str | None = None) -> dict[str, Any]:
        """Parse departure board response.

        Args:
            data: Raw API response

        Returns:
            Parsed departure board data
        """
        # Handle different response structures
        board = data.get("GetStationBoardResult", data)

        location_name = board.get("locationName", "Unknown")
        destination_name = board.get("filterLocationName") or None

        # Extract services
        train_services = board.get("trainServices", {})
        services_list = train_services if isinstance(train_services, list) else train_services.get("service", [])

        if not isinstance(services_list, list):
            services_list = [services_list] if services_list else []

        parsed_services = []
        for service in services_list:
            parsed_service = self._parse_service(service, destination_crs)
            if parsed_service:
                parsed_services.append(parsed_service)

        return {
            "location_name": location_name,
            "destination_name": destination_name,
            "services": parsed_services,
            "generated_at": board.get("generatedAt"),
            "nrcc_messages": board.get("nrccMessages", []),
        }

    def _parse_service(self, service: dict[str, Any], destination_crs: str | None = None) -> dict[str, Any] | None:
        """Parse a single train service.

        Args:
            service: Raw service data

        Returns:
            Parsed service data or None if invalid
        """
        try:
            # Basic service info
            std = service.get("std", "")  # Scheduled departure
            etd = service.get("etd", "")  # Estimated departure
            platform = service.get("platform", "")
            operator_name = service.get("operator", service.get("operatorName", ""))
            service_id = service.get("serviceID", service.get("serviceIdUrlSafe", ""))

            # Determine status
            is_cancelled = etd.lower() in ["cancelled", "canceled"]
            status = STATUS_CANCELLED if is_cancelled else STATUS_ON_TIME

            # Calculate delay
            delay_minutes = 0
            expected_departure = None

            if not is_cancelled and etd and etd != "On time":
                status = STATUS_DELAYED
                expected_departure = etd
                # Try to parse delay from etd if it's a time
                if _TIME_FORMAT_RE.match(etd) and _TIME_FORMAT_RE.match(std):
                    try:
                        # Use a reference date to parse times and handle midnight crossing
                        std_time = datetime.strptime(f"2000-01-01 {std}", "%Y-%m-%d %H:%M")
                        etd_time = datetime.strptime(f"2000-01-01 {etd}", "%Y-%m-%d %H:%M")

                        # Calculate initial time difference
                        time_diff_seconds = (etd_time - std_time).total_seconds()

                        # Handle midnight crossing: if absolute difference > 12 hours, adjust for day boundary
                        if time_diff_seconds < -12 * 3600:
                            # ETD is much earlier in the day, so it's actually next day
                            etd_time += timedelta(days=1)
                        elif time_diff_seconds > 12 * 3600:
                            # ETD is much later in the day, so it's actually previous day
                            etd_time -= timedelta(days=1)

                        delay_minutes = int((etd_time - std_time).total_seconds() / 60)
                    except ValueError:
                        pass

            # Cancellation/delay reason
            cancel_reason = service.get("cancelReason", service.get("delayReason"))
            delay_reason = service.get("delayReason")

            # Destination and calling points
            destination = service.get("destination", [])
            if isinstance(destination, list) and destination:
                destination = destination[0].get("locationName", "")
            elif isinstance(destination, dict):
                destination = destination.get("locationName", "")

            # Subsequent calling points and arrival time
            calling_points = []
            scheduled_arrival = None
            estimated_arrival = None
            subsequent_points = service.get("subsequentCallingPoints", [])
            if isinstance(subsequent_points, list) and subsequent_points:
                calling_point_list = subsequent_points[0].get("callingPoint", [])
                if not isinstance(calling_point_list, list):
                    calling_point_list = [calling_point_list]

                # Build calling points list, truncating at destination if configured
                dest_point = None
                filtered = []
                for cp in calling_point_list:
                    if not cp:
                        continue
                    filtered.append(cp)
                    if destination_crs and cp.get("crs", "").upper() == destination_crs.upper():
                        dest_point = cp
                        break  # Stop collecting stops after the destination

                if dest_point is None and filtered:
                    dest_point = filtered[-1]

                calling_points = [cp.get("locationName", "") for cp in filtered]
                if dest_point:
                    scheduled_arrival = dest_point.get("st")
                    estimated_arrival = dest_point.get("et")

            return {
                "scheduled_departure": std,
                "expected_departure": expected_departure or std,
                "platform": platform,
                "operator": operator_name,
                "service_id": service_id,
                "calling_points": calling_points,
                "delay_minutes": delay_minutes,
                "status": status,
                "is_cancelled": is_cancelled,
                "cancellation_reason": cancel_reason if is_cancelled else None,
                "delay_reason": delay_reason if not is_cancelled else None,
                "scheduled_arrival": scheduled_arrival,
                "estimated_arrival": estimated_arrival or scheduled_arrival,
                "destination": destination,
            }
        except Exception as err:
            _LOGGER.error("Error parsing service: %s", err)
            return None

    def _parse_arrival_board(
        self, data: dict[str, Any], origin_crs: str | None = None
    ) -> dict[str, Any]:
        """Parse arrival board response.

        The board's ``locationName`` is the watched (arrival) station; we
        expose it as ``location_name`` so the coordinator can store it as
        ``destination_name`` (the place trains arrive at). The filter
        station — when supplied — is the train's origin, exposed as
        ``origin_name``.

        Args:
            data: Raw API response
            origin_crs: Origin filter CRS used in the request, if any

        Returns:
            Parsed arrival board data
        """
        board = data.get("GetStationBoardResult", data)

        location_name = board.get("locationName", "Unknown")
        origin_name = board.get("filterLocationName") or None

        train_services = board.get("trainServices", {})
        services_list = (
            train_services
            if isinstance(train_services, list)
            else train_services.get("service", [])
        )

        if not isinstance(services_list, list):
            services_list = [services_list] if services_list else []

        parsed_services = []
        for service in services_list:
            parsed_service = self._parse_arrival_service(service, origin_crs)
            if parsed_service:
                parsed_services.append(parsed_service)

        return {
            "location_name": location_name,
            "origin_name": origin_name,
            "services": parsed_services,
            "generated_at": board.get("generatedAt"),
            "nrcc_messages": board.get("nrccMessages", []),
        }

    def _parse_arrival_service(
        self, service: dict[str, Any], origin_crs: str | None = None
    ) -> dict[str, Any] | None:
        """Parse a single arriving train service.

        Returns the same shape as ``_parse_service`` so downstream code is
        direction-agnostic. ``scheduled_departure``/``expected_departure``
        refer to the train leaving the supplied ``origin_crs`` (taken from
        ``previousCallingPoints``); ``scheduled_arrival``/
        ``estimated_arrival`` refer to its arrival at the watched station.

        Args:
            service: Raw service data
            origin_crs: Filter origin CRS used in the request, if any

        Returns:
            Parsed service data or None if invalid
        """
        try:
            # Arrival at the watched station
            sta = service.get("sta", "")
            eta = service.get("eta", "")
            platform = service.get("platform", "")
            operator_name = service.get("operator", service.get("operatorName", ""))
            service_id = service.get("serviceID", service.get("serviceIdUrlSafe", ""))

            is_cancelled = eta.lower() in ["cancelled", "canceled"]
            status = STATUS_CANCELLED if is_cancelled else STATUS_ON_TIME

            delay_minutes = 0
            estimated_arrival: str | None = None

            if not is_cancelled and eta and eta != "On time":
                status = STATUS_DELAYED
                estimated_arrival = eta
                if _TIME_FORMAT_RE.match(eta) and _TIME_FORMAT_RE.match(sta):
                    try:
                        sta_time = datetime.strptime(f"2000-01-01 {sta}", "%Y-%m-%d %H:%M")
                        eta_time = datetime.strptime(f"2000-01-01 {eta}", "%Y-%m-%d %H:%M")

                        time_diff_seconds = (eta_time - sta_time).total_seconds()

                        if time_diff_seconds < -12 * 3600:
                            eta_time += timedelta(days=1)
                        elif time_diff_seconds > 12 * 3600:
                            eta_time -= timedelta(days=1)

                        delay_minutes = int((eta_time - sta_time).total_seconds() / 60)
                    except ValueError:
                        pass

            cancel_reason = service.get("cancelReason", service.get("delayReason"))
            delay_reason = service.get("delayReason")

            # The arrival board reports the train's overall origin; we keep it
            # as ``origin`` so attributes mirror what a departure entry would
            # carry for the equivalent end of the journey.
            origin_field = service.get("origin", [])
            if isinstance(origin_field, list) and origin_field:
                origin_name = origin_field[0].get("locationName", "")
            elif isinstance(origin_field, dict):
                origin_name = origin_field.get("locationName", "")
            else:
                origin_name = ""

            # The train's eventual terminus — useful context even on an
            # arrival board (it may pass through and continue beyond us).
            destination_field = service.get("destination", [])
            if isinstance(destination_field, list) and destination_field:
                destination_name = destination_field[0].get("locationName", "")
            elif isinstance(destination_field, dict):
                destination_name = destination_field.get("locationName", "")
            else:
                destination_name = ""

            # previousCallingPoints lists the stops the train has already
            # served before reaching us. Find the supplied origin to pull its
            # departure time; fall back to the very first prior stop.
            calling_points: list[str] = []
            scheduled_departure: str | None = None
            expected_departure_time: str | None = None
            previous_points = service.get("previousCallingPoints", [])
            if isinstance(previous_points, list) and previous_points:
                calling_point_list = previous_points[0].get("callingPoint", [])
                if not isinstance(calling_point_list, list):
                    calling_point_list = [calling_point_list]

                origin_point = None
                filtered: list[dict[str, Any]] = []
                # Iterate in chronological order: first item is the journey's
                # earliest stop, last is the one just before our station.
                for cp in calling_point_list:
                    if not cp:
                        continue
                    if origin_crs and not origin_point and cp.get("crs", "").upper() == origin_crs.upper():
                        # Once we hit the supplied origin, keep only it and
                        # subsequent stops in the calling points list —
                        # earlier stops are upstream of the user's journey.
                        origin_point = cp
                        filtered = [cp]
                    else:
                        filtered.append(cp)

                if origin_point is None and filtered:
                    origin_point = filtered[0]

                calling_points = [cp.get("locationName", "") for cp in filtered]
                if origin_point:
                    scheduled_departure = origin_point.get("st")
                    expected_departure_time = origin_point.get("et")

            expected_departure_value = expected_departure_time or scheduled_departure

            return {
                "scheduled_departure": scheduled_departure,
                "expected_departure": expected_departure_value,
                "platform": platform,
                "operator": operator_name,
                "service_id": service_id,
                "calling_points": calling_points,
                "delay_minutes": delay_minutes,
                "status": status,
                "is_cancelled": is_cancelled,
                "cancellation_reason": cancel_reason if is_cancelled else None,
                "delay_reason": delay_reason if not is_cancelled else None,
                "scheduled_arrival": sta,
                "estimated_arrival": estimated_arrival or sta,
                # ``destination`` keeps its existing meaning (the train's
                # final terminus); arrivals additionally expose ``origin`` so
                # cards can distinguish "where it came from" without inspecting
                # calling points.
                "destination": destination_name,
                "origin": origin_name,
            }
        except Exception as err:
            _LOGGER.error("Error parsing arrival service: %s", err)
            return None

    async def validate_station(self, crs_code: str) -> str | None:
        """Validate a station CRS code and return the station name.

        Args:
            crs_code: 3-letter CRS code

        Returns:
            Station name if valid, None otherwise

        Raises:
            InvalidStationError: If station code is invalid
        """
        if not crs_code or len(crs_code) != 3:
            raise InvalidStationError(ERROR_INVALID_STATION)

        _LOGGER.debug("Validating station code: %s", crs_code)

        try:
            # Try to get a simple departure board with minimal rows
            endpoint = f"GetDepartureBoard/{crs_code.upper()}"
            params = {
                "numRows": 1,
            }
            data = await self._request(endpoint, params)

            # Extract station name from response
            board = data.get("GetStationBoardResult", data)
            location_name = board.get("locationName")

            if location_name:
                _LOGGER.debug("Station %s validated: %s", crs_code, location_name)
                return location_name

            raise InvalidStationError(ERROR_INVALID_STATION)

        except (AuthenticationError, RateLimitError):
            # Re-raise auth and rate limit errors
            raise
        except Exception as err:
            _LOGGER.error("Station validation failed for %s: %s", crs_code, err)
            raise InvalidStationError(ERROR_INVALID_STATION) from err

    async def validate_api_key(self) -> bool:
        """Validate the API key by making a test request.

        Returns:
            True if API key is valid

        Raises:
            AuthenticationError: If authentication fails
        """
        _LOGGER.debug("Validating API key")

        try:
            # Make a simple request to validate credentials
            # Use a common station code for testing
            endpoint = "GetDepartureBoard/PAD"  # London Paddington
            params = {
                "numRows": 1,
            }
            await self._request(endpoint, params)
            _LOGGER.debug("API key validated successfully")
            return True

        except AuthenticationError:
            _LOGGER.error("API key validation failed")
            raise
        except Exception as err:
            _LOGGER.error("API key validation error: %s", err)
            raise AuthenticationError(ERROR_AUTH) from err

    async def close(self) -> None:
        """Close the API client and clean up resources.

        This should be called when the API client is no longer needed to ensure
        proper cleanup of the aiohttp ClientSession and prevent resource leaks.
        """
        if self._session and not self._session.closed:
            _LOGGER.debug("Closing aiohttp ClientSession")
            await self._session.close()
        self._session = None
