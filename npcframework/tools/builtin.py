from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import html
import re
import urllib.parse
import urllib.request

from npcframework.core.Runtime_Prompt_Compiler import ToolSpec


def time_now(_: dict) -> Dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "answer": now.strftime("%I:%M %p"),
        "meta": {
            "iso": now.isoformat(),
        },
    }


def add(args: dict) -> Dict[str, Any]:
    # Keep behavior flexible: accept int/float/str that can be casted
    a = float(args["a"])
    b = float(args["b"])
    s = a + b

    # Optional nicety: if it's effectively an int, store that too (doesn't affect canonical)
    as_int = int(s)
    is_integral = abs(s - as_int) < 1e-9

    return {
        "answer": as_int if is_integral else s,   # ✅ canonical
        "meta": {
            "a": a,
            "b": b,
            "is_integral": is_integral,
        },
    }


def _ddg_df_from_recency_days(recency_days: Optional[int]) -> Optional[str]:
    """Map a recency window (days) to DuckDuckGo's df param.

    DuckDuckGo's HTML endpoint supports df=d|w|m|y.
    """
    if recency_days is None:
        return None
    try:
        d = int(recency_days)
    except Exception:
        return None
    if d <= 1:
        return "d"
    if d <= 7:
        return "w"
    if d <= 31:
        return "m"
    return "y"


def _ddg_html_search(query: str, max_results: int = 5, recency_days: Optional[int] = None) -> List[Dict[str, Any]]:
    """Very small, dependency-free DuckDuckGo HTML scrape.

    NOTE: Intended for demo purposes. It may break if DDG changes markup.
    """
    q = (query or "").strip()
    if not q:
        return []

    max_results = max(1, min(int(max_results or 5), 10))

    params = {"q": q}
    df = _ddg_df_from_recency_days(recency_days)
    if df:
        params["df"] = df

    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NPCFramework/1.0)",
            "Accept": "text/html,application/xhtml+xml",
        },
        method="GET",
    )

    # Small timeout so tool calls don't hang the whole turn.
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read(2_000_000)  # cap ~2MB
    html_text = raw.decode("utf-8", errors="replace")

    # Extract titles + urls from DDG HTML results.
    # Example anchor: <a rel="nofollow" class="result__a" href="...">Title</a>
    link_pat = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    results: List[Dict[str, Any]] = []
    for m in link_pat.finditer(html_text):
        href = html.unescape(m.group(1)).strip()
        title_html = m.group(2)
        # Remove nested tags in title (rare but happens)
        title = re.sub(r"<.*?>", "", title_html)
        title = html.unescape(title).strip()

        if not href or not title:
            continue

        results.append({
            "title": title,
            "url": href,
            "source": "duckduckgo",
        })

        if len(results) >= max_results:
            break

    return results


def web_search(args: dict) -> Dict[str, Any]:
    """Search the public web (demo tool).

    Implementation: DuckDuckGo HTML endpoint, no external dependencies.
    """
    query = str(args.get("query", "") or "").strip()
    max_results = int(args.get("max_results", 5) or 5)
    recency_days = args.get("recency_days", None)
    try:
        recency_days = None if recency_days is None else int(recency_days)
    except Exception:
        recency_days = None

    try:
        results = _ddg_html_search(query=query, max_results=max_results, recency_days=recency_days)
        if not results:
            return {
                "answer": "No results found.",
                "meta": {"query": query, "results": []},
            }

        lines = []
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r['title']} — {r['url']}")

        return {
            "answer": "\n".join(lines),
            "meta": {
                "query": query,
                "results": results,
                "recency_days": recency_days,
            },
        }
    except Exception as e:
        return {
            "answer": "Web search failed.",
            "meta": {
                "query": query,
                "error": str(e),
            },
        }



def _decode_ddg_redirect(url: str) -> str:
    """DuckDuckGo result links often look like //duckduckgo.com/l/?uddg=<encoded>.
    Decode them to the real destination URL for downstream fetches.
    """
    try:
        u = url.strip()
        if u.startswith("//"):
            u = "https:" + u
        parsed = urllib.parse.urlparse(u)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return urllib.parse.unquote(qs["uddg"][0])
        return url
    except Exception:
        return url


def fetch_url(args: dict) -> Dict[str, Any]:
    """Fetch a URL and extract readable text (demo tool).

    Notes:
    - Uses stdlib urllib (no dependencies).
    - Strips script/style blocks and collapses whitespace.
    - Hard limits output size for safety.
    """
    url = str(args.get("url", "") or "").strip()
    if not url:
        raise ValueError("missing required field: url")

    max_chars = int(args.get("max_chars", 6000) or 6000)
    timeout_s = int(args.get("timeout_s", 15) or 15)

    # decode DDG redirect links
    url = _decode_ddg_redirect(url)
    if url.startswith("//"):
        url = "https:" + url

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (NPCFramework demo; +https://example.invalid) Python-urllib",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        final_url = getattr(resp, "geturl", lambda: url)()
        content_type = resp.headers.get("Content-Type", "") or ""
        raw = resp.read(2_000_000)  # 2MB safety cap

    # Decode bytes -> text
    encoding = "utf-8"
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, re.I)
    if m:
        encoding = m.group(1).strip()
    try:
        html_text = raw.decode(encoding, errors="replace")
    except Exception:
        html_text = raw.decode("utf-8", errors="replace")

    # Extract title
    title = ""
    mt = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
    if mt:
        title = html.unescape(re.sub(r"\s+", " ", mt.group(1)).strip())

    # Strip scripts/styles and tags
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Trim
    excerpt = cleaned[:max_chars]
    if len(cleaned) > max_chars:
        excerpt += " …"

    return {
        "answer": excerpt,
        "meta": {
            "url": final_url or url,
            "requested_url": url,
            "title": title,
            "content_type": content_type,
            "chars_returned": len(excerpt),
        },
    }



def builtin_toolset(
    allowlist: Optional[List[str]] = None,
) -> Tuple[List[ToolSpec], Dict[str, Any]]:
    all_tools: Dict[str, Tuple[ToolSpec, Any]] = {}

    # --- time_now ---
    all_tools["time_now"] = (
        ToolSpec(
            name="time_now",
            description="Get the current local system time.",
            schema={"type": "object", "properties": {}, "required": []},
        ),
        time_now,
    )

    # --- add ---
    all_tools["add"] = (
        ToolSpec(
            name="add",
            description="Add two numbers.",
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        ),
        add,  # ✅ real function, not lambda
    )

    # --- web_search ---
    all_tools["web_search"] = (
        ToolSpec(
            name="web_search",
            description="Search the public web for information and sources.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "max_results": {"type": "integer", "description": "Max results to return (1-10).", "default": 5},
                    "recency_days": {"type": "integer", "description": "Prefer results within this many days (best-effort)."},
                },
                "required": ["query"],
            },
        ),
        web_search,
    )

    

    # --- fetch_url ---
    all_tools["fetch_url"] = (
        ToolSpec(
            name="fetch_url",
            description="Fetch a webpage by URL and return readable text (for summarization/citation).",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch (http/https)."},
                    "max_chars": {"type": "integer", "description": "Maximum characters of text to return.", "default": 6000},
                    "timeout_s": {"type": "integer", "description": "Request timeout seconds.", "default": 15},
                },
                "required": ["url"],
            },
        ),
        fetch_url,
    )
    if allowlist is None:
            selected = list(all_tools.keys())
    else:
        selected = [name for name in allowlist if name in all_tools]

    available_tools = [all_tools[name][0] for name in selected]
    tool_handlers = {name: all_tools[name][1] for name in selected}

    return available_tools, tool_handlers
