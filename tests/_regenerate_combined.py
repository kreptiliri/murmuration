"""Rebuild `_all_tests_combined.py` — one commented-out copy of the suite.

The combined file exists so the whole suite can be read, or pasted somewhere,
as a single artefact. It is generated, never edited: run

    .venv/bin/python tests/_regenerate_combined.py

after adding or changing a test, and commit the result alongside it.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "_all_tests_combined.py"

# Build order, which is also the order the project was written in.
FILES = ["world.py", "test_geometry.py", "test_separation.py",
         "test_alignment.py", "test_cohesion.py", "test_step.py",
         "test_rollout.py", "test_generate.py", "test_dataset.py"]

RULE = "# " + "=" * 74


def main():
    chunks = []
    for name in FILES:
        body = (HERE / name).read_text().splitlines()
        commented = "\n".join(("# " + line).rstrip() for line in body)
        chunks.append(f"{RULE}\n# {name}\n{RULE}\n{commented}")
    OUT.write_text("\n\n".join(chunks) + "\n")
    print(f"wrote {OUT.name}: {len(FILES)} files, {len(OUT.read_text().splitlines())} lines")


if __name__ == "__main__":
    main()
