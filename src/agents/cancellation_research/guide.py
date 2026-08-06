"""Merchant cancellation guidance: thinkmoney's directory first, search second.

The lookup order is enforced here rather than in a prompt, so no model
preference can reorder it. A merchant the directory covers never reaches the
network; a merchant it does not is searched with the configured provider's
hosted search, and the answer is labelled unverified. Neither path guesses.
"""

import json

from langchain_core.tools import tool

from src.agents.cancellation_research import web_search
from src.agents.cancellation_research.directory import CANCELLATION_DIRECTORY


def _search_available() -> bool:
    """Whether a live search can actually run right now.

    Checked before the degraded answer is written, so "we looked and found
    nothing" is never claimed on a run that had no way to look — no provider
    configured, no credentials, or a provider with no hosted search.
    """
    return web_search.search_available()


def _search_web(merchant: str) -> dict | None:
    """Live-search fallback for merchants the directory does not cover.

    Returns a dict with any of `url`, `steps`, `notice_period`, `gotchas`,
    `results` — the shape `_search_guide` renders — or None when search is
    unavailable, the merchant could not be found, or the reply came back in a
    shape we cannot trust. Every None lands on the degraded answer, which is a
    worse response than a hit but never a wrong one.

    Which backend runs is the configured provider's business, not this tool's:
    see `src/tools/web_search.py:1`.

    Args:
        merchant: The merchant to look up.
    """
    return web_search.search_cancellation(merchant)


def _lookup_directory(merchant: str) -> tuple[dict, str] | None:
    """Find a directory entry by canonical name or alias. Returns (entry, matched)."""
    needle = " ".join(merchant.split()).lower()
    if not needle:
        return None

    for name, entry in CANCELLATION_DIRECTORY.items():
        if needle == name.lower():
            return entry, needle
        if needle in entry["aliases"]:
            return entry, needle

    return None


def _directory_guide(entry: dict, matched_on: str) -> dict:
    # An illustrative entry is plausible rather than researched, so it must not
    # borrow the directory's authority: no verification date, and the caveat
    # travels with the answer.
    verified = not entry["illustrative"]
    return {
        "found": True,
        "merchant": entry["merchant"],
        "source": "directory",
        "matched_on": matched_on,
        "url": entry["url"],
        "steps": list(entry["steps"]),
        "notice_period": entry["notice_period"],
        "gotchas": entry["gotchas"],
        "verified": verified,
        "verified_on": entry["verified_on"] if verified else None,
        "illustrative": entry["illustrative"],
        "note": entry["note"],
        "directory_source": entry["source"],
    }


def _search_guide(merchant: str, result: dict) -> dict:
    return {
        "found": True,
        "merchant": merchant,
        "source": "web_search",
        "url": result.get("url"),
        "steps": list(result.get("steps", [])),
        "notice_period": result.get("notice_period"),
        "gotchas": result.get("gotchas"),
        "verified": False,
        # No verified_on: nobody checked this, and a date here would read as
        # though somebody had.
        "verified_on": None,
        "results": result.get("results", []),
        "caveat": f"{merchant} is not in thinkmoney's cancellation directory, so "
        "this came from a live web search and has not been verified by "
        "us. Treat it as a starting point and confirm the steps on the "
        "merchant's own site before relying on them.",
    }


def _degraded_guide(merchant: str, searched: bool) -> dict:
    """The honest answer for a merchant we cannot help with directly.

    A miss is not an error and never an empty result — the customer still gets
    what we do not know, how to find it themselves, and what the bank can do for
    them regardless of the merchant.
    """
    asked_for = merchant.strip() or "that merchant"
    return {
        "found": False,
        "merchant": merchant,
        # Neither "directory" nor "web_search" would be true here — nothing
        # sourced this answer, and saying otherwise would misattribute it.
        "source": "unavailable",
        "searched": searched,
        "verified": False,
        "verified_on": None,
        "message": f"We do not have cancellation steps for {asked_for} in "
        "thinkmoney's directory"
        + (
            ", and a live search did not turn any up either. "
            if searched
            else ", and live search is not available. "
        )
        + "That means we cannot confirm the exact steps, the notice "
        "period, or whether cancelling online is possible — so "
        "rather than guess, here is how to find out and what we can "
        "do at our end.",
        "how_to_find_it": [
            f"Search the {asked_for} website for 'cancel' or 'manage "
            "subscription' — it is almost always under account or billing "
            "settings.",
            "Check the original confirmation email: the terms, the notice "
            "period and a cancellation link are usually in it.",
            "If there is no online option, ask them in writing (email or their "
            "contact form) and keep a copy — a written request is your evidence "
            "of the date you cancelled.",
        ],
        "your_rights": [
            "If they collect by direct debit, you can cancel the mandate through "
            "us at any time, without their agreement, and we can do it now.",
            "If they bill a card, we can block future payments to them on that "
            "card — that stops the money, though it does not end the contract.",
            "A payment stopping is not the same as a contract ending: until you "
            "cancel with the merchant they can still bill you, chase the balance "
            "or add fees, so do both.",
            "Check the contract for a minimum term or notice period before you "
            "stop paying, so a cancellation does not leave you owing an early "
            "exit fee.",
        ],
    }


@tool
def find_cancellation_guide(merchant: str) -> str:
    """Find out how to cancel a subscription directly with the merchant.

    Use this for any subscription the bank cannot cancel itself — anything billed
    to a card — and whenever the customer asks how to cancel with the merchant.

    thinkmoney's own directory is checked first on every call; it is instant,
    works offline and records when each entry was verified. Only if the merchant
    is not in it is a live search attempted. The response always says where the
    guidance came from, and an unverified answer is labelled as such.

    Args:
        merchant: The merchant name, as the customer or the statement gives it
            (e.g. "Netflix", "Spotify UK", "Adobe Creative Cloud").
    """
    # Directory first, always. A hit returns here and the search path below is
    # never reached — that ordering is the tool's contract, not a heuristic the
    # model gets to reconsider.
    hit = _lookup_directory(merchant)
    if hit is not None:
        entry, matched_on = hit
        return json.dumps(_directory_guide(entry, matched_on))

    # Only now, on a confirmed miss, is a live search worth attempting — and it
    # is optional, so a backend that is absent, empty or broken degrades to the
    # honest answer rather than failing the turn.
    try:
        result = _search_web(merchant)
    except Exception:
        result = None

    if result:
        return json.dumps(_search_guide(merchant, result))

    # Whether a search actually ran changes what we can honestly claim: "we
    # looked and found nothing" is a different statement from "we never looked".
    searched = _search_available()
    return json.dumps(_degraded_guide(merchant, searched=searched))
