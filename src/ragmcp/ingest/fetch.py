"""SSRF-guarded HTTP fetch for `ingest_url`.

Defences: http(s) only, no credentials in the URL, every hostname resolved by us and
refused unless *all* its addresses are globally routable (blocks loopback, private,
link-local incl. cloud metadata, CGNAT, multicast, reserved, IPv4-mapped IPv6 of those),
the connection is pinned to the address we checked (no DNS-rebinding gap), redirects
are followed manually and re-checked each hop, and size/time are capped.
"""

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx2

MAX_REDIRECTS = 3
MAX_BYTES = 10 * 1024 * 1024
TIMEOUT = 20.0
ALLOWED_TYPES = {
    "text/html": ".html", "application/xhtml+xml": ".html", "text/plain": ".txt",
    "text/markdown": ".md", "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


class BlockedURL(ValueError):
    """The URL is refused by policy (SSRF guard, scheme, size, type)."""


Resolver = Callable[[str, int], list[str]]


def system_resolver(host: str, port: int) -> list[str]:
    try:
        return sorted({ai[4][0] for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    except socket.gaierror as e:
        raise BlockedURL(f"cannot resolve host {host!r}") from e


def _is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolver: Resolver = system_resolver) -> tuple[str, str, int, str]:
    """Validate and resolve. Returns (scheme, host, port, pinned_ip) or raises BlockedURL."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise BlockedURL("only http and https URLs are allowed")
    if parts.username or parts.password:
        raise BlockedURL("credentials in URLs are not allowed")
    host = parts.hostname
    if not host:
        raise BlockedURL("URL has no host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as e:
        raise BlockedURL("invalid port") from e
    try:  # literal IPs (incl. odd forms the resolver would normalise) are checked directly
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        addrs = resolver(host, port)
    if not addrs:
        raise BlockedURL(f"cannot resolve host {host!r}")
    bad = [a for a in addrs if not _is_public(a)]
    if bad:
        raise BlockedURL("URL resolves to a private, loopback or otherwise non-public address")
    return parts.scheme, host, port, addrs[0]


def safe_fetch(
    url: str,
    *,
    resolver: Resolver = system_resolver,
    client: httpx2.Client | None = None,
    max_bytes: int = MAX_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    timeout: float = TIMEOUT,
) -> tuple[bytes, str, str]:
    """Fetch `url`. Returns (body, filename_with_extension, final_url)."""
    own = client is None
    client = client or httpx2.Client(timeout=timeout, follow_redirects=False)
    try:
        current = url.strip()
        for _ in range(max_redirects + 1):
            scheme, host, port, ip = check_url(current, resolver)
            parts = urlsplit(current)
            ip_host = f"[{ip}]" if ":" in ip else ip
            default = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
            pinned = urlunsplit((scheme, f"{ip_host}:{port}", parts.path or "/", parts.query, ""))
            headers = {"Host": host if default else f"{host}:{port}", "Accept-Encoding": "identity",
                       "User-Agent": "ragmcp/0.1 (+ingest_url)"}
            with client.stream("GET", pinned, headers=headers, extensions={"sni_hostname": host}) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        raise BlockedURL("redirect without Location")
                    current = urljoin(current, loc)
                    continue
                if resp.status_code != 200:
                    raise BlockedURL(f"server returned HTTP {resp.status_code}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in ALLOWED_TYPES:
                    raise BlockedURL(f"unsupported content type {ctype or 'unknown'!r}")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise BlockedURL(f"response too large (max {max_bytes} bytes)")
                body = bytearray()
                for piece in resp.iter_bytes():
                    body += piece
                    if len(body) > max_bytes:
                        raise BlockedURL(f"response too large (max {max_bytes} bytes)")
                name = (parts.path.rsplit("/", 1)[-1] or host)
                ext = ALLOWED_TYPES[ctype]
                if not name.lower().endswith(ext):
                    name = f"{name}{ext}" if "." not in name else f"{name.rsplit('.', 1)[0]}{ext}"
                return bytes(body), name, current
        raise BlockedURL(f"too many redirects (max {max_redirects})")
    finally:
        if own:
            client.close()
