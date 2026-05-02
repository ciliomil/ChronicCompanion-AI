"""Minimal interactive chat runner."""

from __future__ import annotations

from src.experiments.run_layered_memory import run as run_layered


def main() -> None:
    print("ChronicCompanion prototype chat. Type `exit` to quit.")
    session_id = "cli-session"
    while True:
        query = input("User> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        reply = run_layered(session_id=session_id, query=query)
        print(f"Assistant> {reply}")


if __name__ == "__main__":
    main()
