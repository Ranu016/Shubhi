"""Run a simple multi-turn chat with the local Shubhi model."""

from model.llm_inference import generate_response


def main() -> None:
    print("Shubhi local chat (type /exit to quit)")
    history: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat ended.")
            break

        if prompt.lower() in {"/exit", "/quit"}:
            print("Chat ended.")
            break
        if not prompt:
            continue

        try:
            answer = generate_response(prompt, history=history)
        except RuntimeError as error:
            print(f"Model error: {error}")
            continue

        print(f"Shubhi: {answer}\n")
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )


if __name__ == "__main__":
    main()
