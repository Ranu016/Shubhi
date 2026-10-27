text = "shubhi ai"

# Build a tiny character vocabulary.
characters = sorted(set(text))
char_to_id = {character: index for index, character in enumerate(characters)}
id_to_char = {index: character for character, index in char_to_id.items()}

# Convert characters to numbers, then numbers back to characters.
token_ids = [char_to_id[character] for character in text]
decoded_text = "".join(id_to_char[token_id] for token_id in token_ids)

print("Text:", repr(text))
print("Vocabulary:", characters)
print("Character-to-ID:", char_to_id)
print("Token IDs:", token_ids)
print("Decoded:", repr(decoded_text))

# Make examples where three characters provide context for the next character.
context_size = 3
print("\nContext -> next character:")
for start in range(len(token_ids) - context_size):
    context_ids = token_ids[start : start + context_size]
    next_id = token_ids[start + context_size]

    context = "".join(id_to_char[token_id] for token_id in context_ids)
    next_character = id_to_char[next_id]
    print(f"{context!r} -> {next_character!r}")