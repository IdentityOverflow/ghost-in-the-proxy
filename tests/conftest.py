"""Tests run on code defaults, never on whatever ghost.env/.env configure."""
import os

os.environ["GHOST_NO_CONFIG"] = "1"
