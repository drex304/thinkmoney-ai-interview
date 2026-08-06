"""Prefetched cancellation guidance, keyed by merchant.

This is what makes the cancellation research agent useful without a network call: the answer is
already in the box, so guidance is instant, deterministic and available offline.

The seven real-service entries were researched from vendor documentation on
2026-08-06 and each carries the source it came from. `verified_on` exists because
cancellation flows genuinely change — recording when a fact was captured makes
staleness visible rather than silently wrong, so the agent can say "last checked
August 2026" instead of presenting stale steps as current fact. That research pass
corrected two things that had been written from memory: iCloud does *not* delete
over-quota data (it stops syncing), and Google One has no 60-day read-only window
(the real threshold is two years before content may be deleted).

Two merchants in the transaction corpus are invented — FitLife Gym and the domain
registrar. Their entries are marked `illustrative` and carry no `verified_on`, so
the agent never presents made-up steps as researched fact.

Entries live in this module rather than in `subscriptions.py` so the tool file
stays readable; `find_cancellation_guide` reads from here.
"""

# Keyed by the canonical merchant name. `aliases` carries the lowercased forms a
# lookup might arrive in — including the statement descriptions used in the
# transaction corpus, which are rarely the brand name on its own.
CANCELLATION_DIRECTORY = {
    "Netflix": {
        "merchant": "Netflix",
        "aliases": ["netflix", "netflix.com", "james wilson - shared netflix"],
        "url": "https://www.netflix.com/cancelplan",
        "steps": [
            "Sign in at netflix.com and go to Manage your membership.",
            "Select Cancel.",
            "Select Finish Cancellation to confirm.",
        ],
        "notice_period": (
            "None. Access continues to the end of the current billing period, then the "
            "account cancels automatically."
        ),
        "gotchas": (
            "If no cancellation option appears in your account, you are billed through a "
            "payment partner — Apple, Google, or a TV provider — and must cancel with them. "
            "Netflix cannot do it for you."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": "https://help.netflix.com/en/node/407",
        "note": None,
    },
    "Spotify": {
        "merchant": "Spotify",
        "aliases": ["spotify", "spotify uk", "spotify premium", "spotify ab"],
        "url": "https://www.spotify.com/account/",
        "steps": [
            "Log in to your account page at spotify.com/account.",
            "Under Your plan, select Change plan.",
            "Scroll to Spotify Free and select Cancel Premium.",
            "Continue through to the confirmation message.",
        ],
        "notice_period": (
            "None. Premium runs to the next billing date, then the account reverts to free."
        ),
        "gotchas": (
            "Cancelling during a zero-priced trial switches you to free immediately, not at "
            "period end. Verify success by checking your account page shows the date your "
            "plan changes to Free — if it does not, the cancellation did not complete. "
            "Subscriptions bought through a mobile provider or iTunes must be cancelled "
            "with them."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": "https://support.spotify.com/us/article/cancel-premium/",
        "note": None,
    },
    "Adobe": {
        "merchant": "Adobe",
        "aliases": [
            "adobe",
            "adobe creative cloud",
            "creative cloud",
            "adobe photography plan",
        ],
        "url": "https://account.adobe.com/plans",
        "steps": [
            "Sign in to your Adobe account.",
            "Open Plans.",
            "Select Manage plan.",
            "Select Cancel your plan and follow the prompts.",
        ],
        "notice_period": "None, but the early-termination fee below applies.",
        "gotchas": (
            "Cancel within 14 days of the initial order for a full refund. After 14 days on "
            "an annual plan, an early-termination fee of 50% of the remaining contract "
            "balance applies — cancelling in month nine costs 50% of the remaining three "
            "months. Refunds take 12-14 business days."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": (
            "https://helpx.adobe.com/manage-account/using/"
            "creative-cloud-subscription-terms.html"
        ),
        "note": None,
    },
    "Amazon Prime": {
        "merchant": "Amazon Prime",
        "aliases": ["amazon prime", "prime", "amazon prime membership"],
        "url": (
            "https://www.amazon.co.uk/gp/help/customer/display.html"
            "?nodeId=GTJQ7QZY7QL2HK4Y"
        ),
        "steps": [
            "Go to Your Account.",
            "Open Prime, or Your Memberships and Subscriptions.",
            "Select Manage.",
            "Select Update, Cancel and More, then follow the on-screen instructions.",
        ],
        "notice_period": "None; a 14-day withdrawal right applies from joining.",
        "gotchas": (
            "The refund depends on what you have used: full refund if no order used Prime "
            "benefits, partial if only delivery benefits were used, and none at all if "
            "Prime Video, Music, or Gaming were used."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": (
            "https://www.amazon.co.uk/gp/help/customer/display.html"
            "?nodeId=GTJQ7QZY7QL2HK4Y"
        ),
        "note": None,
    },
    "Disney+": {
        "merchant": "Disney+",
        "aliases": ["disney+", "disney plus", "disneyplus", "disney"],
        "url": "https://www.disneyplus.com/en-gb/commerce/cancel-contract",
        "steps": [
            "Log in at disneyplus.com.",
            "Open Account, then Subscription.",
            "Select Cancel Subscription.",
            "Confirm the cancellation.",
        ],
        "notice_period": "None. Effective at the end of the current subscription term.",
        "gotchas": (
            "Cancelling does not delete your Disney+ or MyDisney account. If you signed up "
            "through a third-party billing partner, cancel through them instead."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": "https://help.disneyplus.com/en-GB/article/disneyplus-en-uk-cancel",
        "note": None,
    },
    "Apple iCloud+": {
        "merchant": "Apple iCloud+",
        "aliases": ["apple icloud+", "icloud+", "icloud", "apple icloud", "apple"],
        "url": "https://support.apple.com/en-us/108318",
        "steps": [
            "Open Settings and tap your name.",
            "Tap Subscriptions.",
            "Tap iCloud+ under Active.",
            "Tap Cancel Subscription, or See All Plans to downgrade instead.",
        ],
        "notice_period": "Takes effect after the current billing period ends.",
        "gotchas": (
            "Your data is not deleted if you end up over quota — but iCloud stops syncing "
            "and backups stop completing until you free up space or increase capacity. "
            "Download or remove anything above your new allowance before you downgrade."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": "https://support.apple.com/en-us/108318",
        "note": None,
    },
    "Google One": {
        "merchant": "Google One",
        "aliases": ["google one", "google one storage", "google storage"],
        "url": "https://one.google.com",
        "steps": [
            "Go to one.google.com.",
            "Open Settings.",
            "Select Cancel membership.",
            "Confirm the cancellation.",
        ],
        "notice_period": "None stated.",
        "gotchas": (
            "If you are over the free quota after cancelling, Gmail stops sending and "
            "receiving and messages bounce back to the sender, Drive stops syncing and "
            "uploading, nobody can edit or copy your files, and Photos stops backing up. "
            "Content may be deleted after 2 years over quota."
        ),
        "verified_on": "2026-08-06",
        "illustrative": False,
        "source": "https://support.google.com/googleone/answer/9056360?hl=en",
        "note": None,
    },
    "FitLife Gym": {
        "merchant": "FitLife Gym",
        "aliases": ["fitlife gym", "fitlife", "fit life gym"],
        "url": "https://fitlife.co.uk/account/membership",
        "steps": [
            "Log in to your FitLife account.",
            "Open Account, then Membership.",
            "Select Cancel membership and submit the written notice form.",
            "Keep the confirmation email — it is the proof your notice period started.",
        ],
        "notice_period": "30 days' written notice; a minimum term may still apply.",
        "gotchas": (
            "The 30 days run from when the notice is received, so one more payment is "
            "usually collected after you cancel. If you are inside a minimum term, the "
            "remaining months may still be owed."
        ),
        "verified_on": None,
        "illustrative": True,
        "source": None,
        "note": (
            "Illustrative entry — FitLife Gym is a fictional merchant invented for the "
            "transaction fixture, so these steps are plausible rather than researched."
        ),
    },
    "Namecheap": {
        "merchant": "Namecheap",
        "aliases": [
            "namecheap",
            "namecheap domain renewal",
            "domain renewal",
            "domain registrar",
        ],
        "url": "https://www.namecheap.com/myaccount/login/",
        "steps": [
            "Log in to your registrar account.",
            "Open Domain List and select the domain.",
            "Turn off auto-renew.",
            "Let the domain lapse at expiry, or transfer it away if you want to keep it.",
        ],
        "notice_period": "None — turn auto-renew off before the renewal date.",
        "gotchas": (
            "Turning off auto-renew is not the same as cancelling the domain: you keep it "
            "until it expires, and once it does expire the name can be registered by "
            "somebody else. Any email or website on that domain stops working."
        ),
        "verified_on": None,
        "illustrative": True,
        "source": None,
        "note": (
            "Illustrative entry — the domain registrar behind this transaction is a stand-in "
            "invented for the fixture, so these steps are plausible rather than researched."
        ),
    },
}


REAL_MERCHANTS = tuple(
    name for name, entry in CANCELLATION_DIRECTORY.items() if not entry["illustrative"]
)

FICTIONAL_MERCHANTS = tuple(
    name for name, entry in CANCELLATION_DIRECTORY.items() if entry["illustrative"]
)
