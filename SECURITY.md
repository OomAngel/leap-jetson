# Security policy

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/OomAngel/leap-jetson/security/advisories/new)
rather than a public issue. This is a one-person project: expect a first reply within 14 days.

## Scope

leap-jetson runs locally: it reads a USB device the user owns and loads models the user downloaded.
It opens no network ports. The fallback in `fallback_v2_box64/` runs third-party software under an
emulator, and that software's own server listens on localhost; issues in that software belong to its
vendor.
